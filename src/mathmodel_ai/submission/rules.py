from __future__ import annotations

import fnmatch
import re
import unicodedata
from datetime import UTC, datetime
from io import BytesIO
from pathlib import PurePosixPath
from uuid import UUID

from PIL import ExifTags, Image, UnidentifiedImageError
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from mathmodel_ai.files.storage import FileStore
from mathmodel_ai.schemas.submission import (
    CompetitionProfile,
    CompetitionRule,
    DeadlineMode,
    DeadlineStatus,
    InclusionPolicy,
    PageScope,
    ProfileVerificationStatus,
    RuleResult,
    RuleResultStatus,
    RuleSeverity,
    RuleType,
    SubmissionArtifact,
    SubmissionArtifactRole,
    SubmissionCandidate,
)

_ABSOLUTE_PATHS = re.compile(
    r"(?i)(?:"
    r"file:/+(?:[A-Z]:/|/)?[^\s<>'\"]+|"
    r"(?<![A-Za-z0-9])[A-Z]:[\\/][^\s<>'\"]+|"
    r"(?:\\\\|(?<!:)//)[^\\/\s]+[\\/][^\s<>'\"]+|"
    r"/(?:home|Users|tmp)/[^\s<>'\"]+"
    r")"
)
_EMAIL = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
_IDENTITY_LABEL = re.compile(
    r"(?i)\b(?:author|advisor|student\s*id|school|university|organization|phone)\s*[:\uFF1A]"
)
_SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|client[_-]?secret|token|password|"
        r"secret|license[_-]?key|database[_-]?url)\s*[:=]\s*[^\s,;]{6,}"
    ),
    re.compile(
        r"(?i)\b(?:postgres(?:ql)?|mysql|mariadb|mongodb(?:\+srv)?|redis)"
        r"(?:\+\w+)?://[^\s/@:]+:[^\s/@]+@"
    ),
]
_MOCK_INDICATOR = re.compile(
    r"(?i)(?:[\"']?\bis_mock\b[\"']?\s*[:=]\s*true\b|\bTEST_FIXTURE\b|\bmock_result\b)"
)
_OBFUSCATED_EMAIL = re.compile(r"(?i)[^\s@]*[\u0370-\u03ff\u0400-\u04ff][^\s@]*@[^\s@]+")
_FORBIDDEN_PARTS = {
    ".git",
    ".env",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    ".DS_Store",
    "Thumbs.db",
}
_FORBIDDEN_FILE_NAMES = {
    ".coverage",
    "coverage.xml",
    "AGENTS.md",
}
_FORBIDDEN_SUFFIXES = {".log", ".sqlite", ".sqlite3", ".db", ".pyc", ".pyo", ".map"}
_INTERNAL_PATH_MARKERS = {
    "adversarial-report",
    "security-audit",
    "exec-plans",
    "internal-prompt",
    "mock_result",
    "fixture_reference",
    "test_profile",
}


def _result(
    rule: CompetitionRule, status: RuleResultStatus, message: str, *refs: str
) -> RuleResult:
    return RuleResult(
        rule_id=rule.rule_id,
        rule_type=rule.rule_type,
        severity=rule.severity,
        status=status,
        message=message,
        evidence_refs=list(refs),
    )


class DeadlineEvaluator:
    @staticmethod
    def evaluate(
        deadline: datetime | None, now: datetime | None = None
    ) -> tuple[DeadlineStatus, DeadlineMode]:
        if deadline is None:
            return DeadlineStatus.UNKNOWN, DeadlineMode.UNKNOWN
        if deadline.tzinfo is None:
            raise ValueError("deadline must be timezone-aware")
        current = now or datetime.now(UTC)
        hours = (deadline - current).total_seconds() / 3600
        if hours < 0:
            return DeadlineStatus.EXPIRED, DeadlineMode.SUBMISSION_MODE
        status = DeadlineStatus.NEAR_DEADLINE if hours < 3 else DeadlineStatus.OPEN
        if hours < 1:
            mode = DeadlineMode.SUBMISSION_MODE
        elif hours < 3:
            mode = DeadlineMode.MODEL_FREEZE
        elif hours < 6:
            mode = DeadlineMode.FINALIZATION
        elif hours <= 12:
            mode = DeadlineMode.PRIORITY
        else:
            mode = DeadlineMode.NORMAL
        return status, mode


class SubmissionSecurityScanner:
    """Mandatory package scanner; deadline mode never weakens these checks."""

    def scan(
        self,
        candidate: SubmissionCandidate,
        artifact_bytes: dict[UUID, bytes],
    ) -> list[tuple[str, str]]:
        findings: list[tuple[str, str]] = []
        candidate_text = _normalize_text(
            "\n".join(
                [
                    candidate.paper_text,
                    *(
                        f"{key}: {value}"
                        for key, value in candidate.paper_metadata.items()
                        if str(value).strip()
                    ),
                ]
            )
        )
        if _ABSOLUTE_PATHS.search(candidate_text):
            findings.append(("ABSOLUTE_PATH_LEAK", "paper content or metadata"))
        if any(pattern.search(candidate_text) for pattern in _SECRET_PATTERNS):
            findings.append(("SECRET_LEAK", "paper content or metadata"))
        if _MOCK_INDICATOR.search(candidate_text):
            findings.append(("MOCK_INDICATOR", "paper content or metadata"))
        paths = [item.relative_path for item in candidate.artifacts]
        if len(paths) != len({_canonical_path(item) for item in paths}):
            findings.append(("DUPLICATE_PATH", "submission contains duplicate paths"))
        findings.extend(self.scan_artifacts(candidate.artifacts, artifact_bytes))
        return sorted(set(findings))

    def scan_artifacts(
        self,
        artifacts: list[SubmissionArtifact],
        artifact_bytes: dict[UUID, bytes],
    ) -> list[tuple[str, str]]:
        findings: list[tuple[str, str]] = []
        for artifact in artifacts:
            path = PurePosixPath(artifact.relative_path)
            canonical_parts = [
                unicodedata.normalize("NFKC", part).casefold() for part in path.parts
            ]
            filename = canonical_parts[-1]
            if (
                any(
                    part in {item.casefold() for item in _FORBIDDEN_PARTS} or part.startswith(".")
                    for part in canonical_parts
                )
                or filename in {item.casefold() for item in _FORBIDDEN_FILE_NAMES}
                or PurePosixPath(filename).suffix in _FORBIDDEN_SUFFIXES
                or any(marker in "/".join(canonical_parts) for marker in _INTERNAL_PATH_MARKERS)
            ):
                findings.append(("FORBIDDEN_FILE", artifact.relative_path))
            data = artifact_bytes.get(artifact.artifact_id)
            if data is None:
                findings.append(("MISSING_ARTIFACT_BYTES", artifact.relative_path))
                continue
            if len(data) != artifact.size_bytes:
                findings.append(("ARTIFACT_SIZE_MISMATCH", artifact.relative_path))
            text = _decode_scannable(artifact, data)
            if text is None:
                continue
            text = _normalize_text(text)
            if _ABSOLUTE_PATHS.search(text):
                findings.append(("ABSOLUTE_PATH_LEAK", artifact.relative_path))
            if any(pattern.search(text) for pattern in _SECRET_PATTERNS):
                findings.append(("SECRET_LEAK", artifact.relative_path))
            if _MOCK_INDICATOR.search(text):
                findings.append(("MOCK_INDICATOR", artifact.relative_path))
            if artifact.role is SubmissionArtifactRole.PAPER_PDF and _pdf_has_attachments(data):
                findings.append(("EMBEDDED_ATTACHMENT", artifact.relative_path))
        return sorted(set(findings))

    def anonymity_findings(
        self,
        candidate: SubmissionCandidate,
        artifact_bytes: dict[UUID, bytes],
        prohibited_terms: list[str],
    ) -> list[str]:
        text_parts = [
            candidate.paper_text,
            *(
                f"{key}: {value}"
                for key, value in candidate.paper_metadata.items()
                if str(value).strip()
            ),
        ]
        for artifact in candidate.artifacts:
            data = artifact_bytes.get(artifact.artifact_id)
            if data is None:
                continue
            text = _decode_scannable(artifact, data)
            if text is not None:
                text_parts.append(text)
        return _identity_findings("\n".join(text_parts), prohibited_terms)

    def anonymity_artifact_findings(
        self,
        artifacts: list[SubmissionArtifact],
        artifact_bytes: dict[UUID, bytes],
        prohibited_terms: list[str],
    ) -> list[str]:
        text_parts: list[str] = []
        for artifact in artifacts:
            data = artifact_bytes.get(artifact.artifact_id)
            if data is None:
                continue
            text = _decode_scannable(artifact, data)
            if text is not None:
                text_parts.append(text)
        return _identity_findings("\n".join(text_parts), prohibited_terms)


def _identity_findings(text: str, prohibited_terms: list[str]) -> list[str]:
    text = _normalize_text(text)
    lowered = text.casefold()
    findings = [term for term in prohibited_terms if _normalize_text(term).casefold() in lowered]
    if _EMAIL.search(text):
        findings.append("email address")
    if _IDENTITY_LABEL.search(text):
        findings.append("identity label")
    if _ABSOLUTE_PATHS.search(text):
        findings.append("local filesystem path")
    if _OBFUSCATED_EMAIL.search(text):
        findings.append("obfuscated identity requires human review")
    return sorted(set(findings))


class RuleEngine:
    def __init__(self, store: FileStore) -> None:
        self._store = store
        self.security = SubmissionSecurityScanner()

    def read_artifact_bytes(self, artifacts: list[SubmissionArtifact]) -> dict[UUID, bytes]:
        return {
            artifact.artifact_id: self._store.read_bytes(artifact.storage_key)
            for artifact in artifacts
        }

    def evaluate(
        self,
        profile: CompetitionProfile,
        candidate: SubmissionCandidate,
        *,
        artifact_bytes: dict[UUID, bytes] | None = None,
        now: datetime | None = None,
    ) -> list[RuleResult]:
        if (candidate.competition_profile_id, candidate.competition_profile_version) != (
            profile.profile_id,
            profile.version,
        ):
            raise ValueError("candidate is bound to a different competition profile version")
        payloads = artifact_bytes or self.read_artifact_bytes(candidate.artifacts)
        return [
            self._evaluate_rule(rule, profile, candidate, payloads, now) for rule in profile.rules
        ]

    def _evaluate_rule(
        self,
        rule: CompetitionRule,
        profile: CompetitionProfile,
        candidate: SubmissionCandidate,
        payloads: dict[UUID, bytes],
        now: datetime | None,
    ) -> RuleResult:
        if rule.verification_status in {
            ProfileVerificationStatus.UNVERIFIED,
            ProfileVerificationStatus.UNKNOWN,
            ProfileVerificationStatus.CONFLICT,
        } or (
            rule.severity is RuleSeverity.BLOCKING
            and profile.verification_status
            in {
                ProfileVerificationStatus.UNVERIFIED,
                ProfileVerificationStatus.UNKNOWN,
                ProfileVerificationStatus.CONFLICT,
            }
        ):
            return _result(
                rule,
                RuleResultStatus.HUMAN_REVIEW,
                "rule provenance is not verified",
                rule.source_ref,
            )
        handlers = {
            RuleType.PAGE_LIMIT: self._page_limit,
            RuleType.FILE_FORMAT: self._file_format,
            RuleType.FILE_COUNT: self._file_count,
            RuleType.FILE_SIZE: self._file_size,
            RuleType.FILENAME: self._filename,
            RuleType.ANONYMITY: self._anonymity,
            RuleType.REQUIRED_SECTION: self._required_section,
            RuleType.PROHIBITED_CONTENT: self._prohibited_content,
            RuleType.CODE_SUBMISSION: self._content_policy,
            RuleType.DATA_SUBMISSION: self._content_policy,
            RuleType.AI_DISCLOSURE: self._ai_disclosure,
            RuleType.DEADLINE: self._deadline,
        }
        handler = handlers.get(rule.rule_type)
        if handler is None:
            status = (
                RuleResultStatus.HUMAN_REVIEW
                if rule.severity is RuleSeverity.BLOCKING
                else RuleResultStatus.NOT_EVALUABLE
            )
            return _result(rule, status, "rule type is not deterministically evaluable")
        return handler(rule, profile, candidate, payloads, now)

    def _page_limit(
        self,
        rule: CompetitionRule,
        profile: CompetitionProfile,
        candidate: SubmissionCandidate,
        payloads: dict[UUID, bytes],
        _now: datetime | None,
    ) -> RuleResult:
        scope = PageScope(rule.parameters.get("scope", profile.page_rules.scope.value))
        if scope is not PageScope.TOTAL:
            return _result(
                rule,
                RuleResultStatus.HUMAN_REVIEW,
                "page scope cannot be separated reliably from the final PDF",
            )
        maximum = int(rule.parameters.get("max_pages", profile.page_rules.max_pages or 0))
        if maximum <= 0:
            return _result(rule, RuleResultStatus.NOT_EVALUABLE, "page limit is unspecified")
        papers = [
            item for item in candidate.artifacts if item.role is SubmissionArtifactRole.PAPER_PDF
        ]
        if len(papers) != 1:
            return _result(rule, RuleResultStatus.FAIL, "formal paper PDF is missing or ambiguous")
        try:
            actual_pages = len(
                PdfReader(BytesIO(payloads[papers[0].artifact_id]), strict=True).pages
            )
        except (KeyError, PdfReadError, ValueError, TypeError):
            return _result(rule, RuleResultStatus.FAIL, "formal paper PDF cannot be parsed")
        if actual_pages != candidate.page_count:
            return _result(
                rule, RuleResultStatus.FAIL, "candidate page count disagrees with PDF bytes"
            )
        if actual_pages > maximum:
            return _result(
                rule,
                RuleResultStatus.FAIL,
                f"PDF has {actual_pages} pages; maximum is {maximum}",
            )
        return _result(rule, RuleResultStatus.PASS, "real PDF page count is within limit")

    @staticmethod
    def _file_format(
        rule: CompetitionRule,
        _profile: CompetitionProfile,
        candidate: SubmissionCandidate,
        payloads: dict[UUID, bytes],
        _now: datetime | None,
    ) -> RuleResult:
        role = SubmissionArtifactRole(str(rule.parameters.get("role")))
        expected = str(rule.parameters.get("mime_type", ""))
        matches = [item for item in candidate.artifacts if item.role is role]
        if not matches:
            return _result(rule, RuleResultStatus.FAIL, f"required {role.value} artifact is absent")
        for artifact in matches:
            data = payloads.get(artifact.artifact_id, b"")
            actual = detect_mime(data)
            if artifact.mime_type != expected or actual != expected:
                return _result(
                    rule,
                    RuleResultStatus.FAIL,
                    f"{artifact.relative_path} is {actual or 'unknown'}, expected {expected}",
                )
            if expected == "application/pdf":
                try:
                    if len(PdfReader(BytesIO(data), strict=True).pages) < 1:
                        raise ValueError("empty PDF")
                except (PdfReadError, ValueError, TypeError, KeyError):
                    return _result(rule, RuleResultStatus.FAIL, "paper PDF failed strict parsing")
        return _result(rule, RuleResultStatus.PASS, "file signatures match declared MIME types")

    @staticmethod
    def _file_count(
        rule: CompetitionRule,
        _profile: CompetitionProfile,
        candidate: SubmissionCandidate,
        _payloads: dict[UUID, bytes],
        _now: datetime | None,
    ) -> RuleResult:
        maximum = int(rule.parameters.get("maximum", 0))
        status = (
            RuleResultStatus.PASS
            if maximum > 0 and len(candidate.artifacts) <= maximum
            else RuleResultStatus.FAIL
        )
        return _result(rule, status, f"candidate contains {len(candidate.artifacts)} formal files")

    @staticmethod
    def _file_size(
        rule: CompetitionRule,
        profile: CompetitionProfile,
        candidate: SubmissionCandidate,
        _payloads: dict[UUID, bytes],
        _now: datetime | None,
    ) -> RuleResult:
        maximum = int(rule.parameters.get("maximum", profile.file_rules.max_file_size_bytes or 0))
        oversized = [
            item.relative_path for item in candidate.artifacts if item.size_bytes > maximum
        ]
        if maximum <= 0:
            return _result(rule, RuleResultStatus.NOT_EVALUABLE, "file size limit is unspecified")
        if oversized:
            return _result(rule, RuleResultStatus.FAIL, "oversized files: " + ", ".join(oversized))
        return _result(rule, RuleResultStatus.PASS, "all files are within size limits")

    @staticmethod
    def _filename(
        rule: CompetitionRule,
        profile: CompetitionProfile,
        candidate: SubmissionCandidate,
        _payloads: dict[UUID, bytes],
        _now: datetime | None,
    ) -> RuleResult:
        role = SubmissionArtifactRole(str(rule.parameters.get("role")))
        exact = str(rule.parameters.get("exact", profile.naming_rules.paper_filename or ""))
        files = [
            PurePosixPath(item.relative_path).name
            for item in candidate.artifacts
            if item.role is role
        ]
        compare = (lambda value: value) if profile.naming_rules.case_sensitive else str.casefold
        if len(files) != 1:
            return _result(rule, RuleResultStatus.FAIL, "expected exactly one matching formal file")
        if exact and compare(files[0]) != compare(exact):
            return _result(rule, RuleResultStatus.FAIL, f"expected exact filename {exact}")
        prefix = str(rule.parameters.get("prefix", profile.naming_rules.prefix or ""))
        suffix = str(rule.parameters.get("suffix", profile.naming_rules.suffix or ""))
        if prefix and not compare(files[0]).startswith(compare(prefix)):
            return _result(rule, RuleResultStatus.FAIL, f"filename must start with {prefix}")
        if suffix and not compare(files[0]).endswith(compare(suffix)):
            return _result(rule, RuleResultStatus.FAIL, f"filename must end with {suffix}")
        if not re.fullmatch(profile.naming_rules.allowed_pattern, files[0]):
            return _result(rule, RuleResultStatus.FAIL, "filename uses forbidden characters")
        return _result(rule, RuleResultStatus.PASS, "filename matches profile")

    def _anonymity(
        self,
        rule: CompetitionRule,
        profile: CompetitionProfile,
        candidate: SubmissionCandidate,
        payloads: dict[UUID, bytes],
        _now: datetime | None,
    ) -> RuleResult:
        required = bool(rule.parameters.get("required", profile.anonymous_rules.required))
        if not required:
            return _result(rule, RuleResultStatus.PASS, "anonymity is not required")
        findings = self.security.anonymity_findings(
            candidate, payloads, profile.anonymous_rules.prohibited_terms
        )
        if findings:
            return _result(rule, RuleResultStatus.FAIL, "anonymity leak: " + ", ".join(findings))
        return _result(rule, RuleResultStatus.PASS, "no configured identity leak was detected")

    @staticmethod
    def _required_section(
        rule: CompetitionRule,
        profile: CompetitionProfile,
        candidate: SubmissionCandidate,
        _payloads: dict[UUID, bytes],
        _now: datetime | None,
    ) -> RuleResult:
        required = set(rule.parameters.get("sections", profile.required_sections))
        missing = required - set(candidate.section_types)
        if missing:
            return _result(
                rule, RuleResultStatus.FAIL, "missing sections: " + ", ".join(sorted(missing))
            )
        return _result(rule, RuleResultStatus.PASS, "required sections exist")

    @staticmethod
    def _prohibited_content(
        rule: CompetitionRule,
        _profile: CompetitionProfile,
        candidate: SubmissionCandidate,
        _payloads: dict[UUID, bytes],
        _now: datetime | None,
    ) -> RuleResult:
        terms = [str(item) for item in rule.parameters.get("terms", [])]
        found = [item for item in terms if item.casefold() in candidate.paper_text.casefold()]
        if found:
            return _result(rule, RuleResultStatus.FAIL, "prohibited content: " + ", ".join(found))
        return _result(rule, RuleResultStatus.PASS, "prohibited content was not found")

    @staticmethod
    def _content_policy(
        rule: CompetitionRule,
        profile: CompetitionProfile,
        candidate: SubmissionCandidate,
        _payloads: dict[UUID, bytes],
        _now: datetime | None,
    ) -> RuleResult:
        role = (
            SubmissionArtifactRole.CODE
            if rule.rule_type is RuleType.CODE_SUBMISSION
            else SubmissionArtifactRole.DATA
        )
        policy = InclusionPolicy(str(rule.parameters.get("policy")))
        present = any(item.role is role for item in candidate.artifacts)
        if policy is InclusionPolicy.REQUIRED and not present:
            return _result(rule, RuleResultStatus.FAIL, f"required {role.value.lower()} is missing")
        if policy is InclusionPolicy.PROHIBITED and present:
            return _result(
                rule, RuleResultStatus.FAIL, f"prohibited {role.value.lower()} is present"
            )
        content_rules = (
            profile.code_submission_rules
            if role is SubmissionArtifactRole.CODE
            else profile.data_submission_rules
        )
        if (
            present
            and content_rules.readme_required
            and not any(item.role is SubmissionArtifactRole.README for item in candidate.artifacts)
        ):
            return _result(rule, RuleResultStatus.FAIL, "required README is missing")
        if present and content_rules.entrypoint_required:
            configured = str(profile.special_rules.get("code_entrypoint", "code/main.py"))
            if not any(item.relative_path == configured for item in candidate.artifacts):
                return _result(
                    rule, RuleResultStatus.FAIL, f"required entrypoint {configured} is missing"
                )
        return _result(rule, RuleResultStatus.PASS, f"{role.value.lower()} policy is satisfied")

    @staticmethod
    def _ai_disclosure(
        rule: CompetitionRule,
        profile: CompetitionProfile,
        candidate: SubmissionCandidate,
        _payloads: dict[UUID, bytes],
        _now: datetime | None,
    ) -> RuleResult:
        policy = profile.ai_disclosure_rules.policy
        if policy.value == "UNKNOWN":
            return _result(rule, RuleResultStatus.HUMAN_REVIEW, "AI disclosure rule is unknown")
        if policy.value == "PROHIBITED":
            return _result(
                rule,
                RuleResultStatus.FAIL,
                "profile prohibits AI use but this workflow is AI-assisted",
            )
        required_text = profile.ai_disclosure_rules.required_text
        if policy.value in {"REQUIRED", "ALLOWED_WITH_CONDITIONS"} and (
            required_text is None or required_text.casefold() not in candidate.paper_text.casefold()
        ):
            return _result(rule, RuleResultStatus.FAIL, "required AI disclosure is absent")
        return _result(
            rule, RuleResultStatus.PASS, "AI disclosure policy is deterministically satisfied"
        )

    @staticmethod
    def _deadline(
        rule: CompetitionRule,
        profile: CompetitionProfile,
        _candidate: SubmissionCandidate,
        _payloads: dict[UUID, bytes],
        now: datetime | None,
    ) -> RuleResult:
        status, mode = DeadlineEvaluator.evaluate(profile.deadline, now)
        if status is DeadlineStatus.UNKNOWN:
            return _result(rule, RuleResultStatus.HUMAN_REVIEW, "competition deadline is unknown")
        if status is DeadlineStatus.EXPIRED:
            return _result(rule, RuleResultStatus.FAIL, "competition deadline has expired")
        return _result(
            rule, RuleResultStatus.PASS, f"deadline is {status.value}; mode={mode.value}"
        )


def detect_mime(data: bytes) -> str | None:
    if data.startswith(b"%PDF-"):
        return "application/pdf"
    if data.startswith(b"PK\x03\x04"):
        return "application/zip"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith((b"\xff\xd8\xff",)):
        return "image/jpeg"
    if b"\x00" not in data[:4096]:
        return "text/plain"
    return None


def mime_matches(declared: str, data: bytes) -> bool:
    actual = detect_mime(data)
    if declared.startswith("text/") or declared in {
        "application/x-tex",
        "application/x-bibtex",
        "application/json",
    }:
        return actual == "text/plain"
    return actual == declared


def _decode_scannable(artifact: SubmissionArtifact, data: bytes) -> str | None:
    if artifact.role is SubmissionArtifactRole.PAPER_PDF:
        return _pdf_scannable_text(data)
    if artifact.mime_type.startswith("text/") or artifact.mime_type in {
        "application/json",
        "application/x-tex",
        "application/x-bibtex",
    }:
        return data.decode("utf-8", errors="replace")
    if artifact.mime_type.startswith("image/"):
        return _image_metadata_text(data)
    return None


def _pdf_scannable_text(data: bytes) -> str | None:
    try:
        reader = PdfReader(BytesIO(data), strict=True)
        parts = [
            *(
                f"{key}: {value}"
                for key, value in (reader.metadata or {}).items()
                if str(value).strip()
            ),
            *(page.extract_text() or "" for page in reader.pages),
        ]
        for filename, payloads in reader.attachments.items():
            parts.append(f"embedded attachment: {filename}")
            parts.extend(payload.decode("utf-8", errors="replace") for payload in payloads)
        return "\n".join(parts)
    except (PdfReadError, ValueError, TypeError, KeyError, OSError):
        return None


def _pdf_has_attachments(data: bytes) -> bool:
    try:
        return bool(PdfReader(BytesIO(data), strict=True).attachments)
    except (PdfReadError, ValueError, TypeError, KeyError, OSError):
        return False


def _image_metadata_text(data: bytes) -> str | None:
    try:
        with Image.open(BytesIO(data)) as image:
            parts: list[str] = []
            for info_key, value in image.info.items():
                if isinstance(value, bytes):
                    value = value.decode("utf-8", errors="replace")
                parts.append(f"{info_key}: {value}")
            for exif_key, value in image.getexif().items():
                parts.append(f"{ExifTags.TAGS.get(exif_key, str(exif_key))}: {value}")
            return "\n".join(parts)
    except (OSError, TypeError, UnidentifiedImageError, ValueError):
        return None


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return "".join(character for character in normalized if unicodedata.category(character) != "Cf")


def _canonical_path(value: str) -> str:
    return unicodedata.normalize("NFC", value.replace("\\", "/")).casefold()


def is_allowed_path(path: str, profile: CompetitionProfile) -> bool:
    normalized = path.replace("\\", "/")
    if any(fnmatch.fnmatchcase(normalized, item) for item in profile.prohibited_submission_files):
        return False
    return any(fnmatch.fnmatchcase(normalized, item) for item in profile.allowed_submission_files)

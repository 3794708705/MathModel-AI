import base64
import binascii
import os
import re
import secrets as secure_random
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from datetime import UTC, datetime

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydantic import SecretStr
from sqlalchemy.orm import Session, sessionmaker

from mathmodel_ai.core.errors import ConfigurationError
from mathmodel_ai.db.models import EncryptedSecretRecordModel
from mathmodel_ai.db.session import session_scope

_ENV_REF = re.compile(r"^env:([A-Z_][A-Z0-9_]{1,126})$")
_SECRET_REF = re.compile(r"^secret:(provider/[a-z0-9][a-z0-9._-]{1,99}/[a-f0-9]{12})$")
CredentialSupplier = Callable[[], str]


def credential_supplier(
    *, api_key: str | None, supplier: CredentialSupplier | None
) -> CredentialSupplier:
    if supplier is not None:
        return supplier
    if api_key is None or not api_key.strip():
        raise ConfigurationError("provider credential is not configured")
    return lambda: api_key


class BaseSecretResolver(ABC):
    @abstractmethod
    def resolve(self, credential_ref: str) -> SecretStr:
        """Resolve a reference without exposing it through configuration objects."""

    @abstractmethod
    def is_configured(self, credential_ref: str | None) -> bool:
        """Return only whether a non-empty value exists."""


class EnvironmentSecretResolver(BaseSecretResolver):
    def __init__(self, environment: Mapping[str, str] | None = None) -> None:
        self._environment = environment if environment is not None else os.environ

    @staticmethod
    def variable_name(credential_ref: str) -> str:
        match = _ENV_REF.fullmatch(credential_ref)
        if match is None:
            raise ConfigurationError("only env:NAME credential references are supported")
        return match.group(1)

    def resolve(self, credential_ref: str) -> SecretStr:
        variable = self.variable_name(credential_ref)
        value = self._environment.get(variable)
        if value is None or not value.strip():
            raise ConfigurationError(f"credential reference env:{variable} is not configured")
        return SecretStr(value)

    def is_configured(self, credential_ref: str | None) -> bool:
        if credential_ref is None:
            return False
        variable = self.variable_name(credential_ref)
        value = self._environment.get(variable)
        return value is not None and bool(value.strip())


class BaseSecretStore(ABC):
    @property
    @abstractmethod
    def available(self) -> bool:
        """Return whether this process can encrypt and decrypt values."""

    @abstractmethod
    def put(self, secret_id: str, value: SecretStr) -> None:
        """Encrypt and persist one secret value."""

    @abstractmethod
    def resolve(self, secret_id: str) -> SecretStr:
        """Decrypt one secret value inside the server boundary."""

    @abstractmethod
    def contains(self, secret_id: str) -> bool:
        """Return only whether the secret exists."""

    @abstractmethod
    def delete(self, secret_id: str) -> bool:
        """Delete one encrypted secret if present."""


class EncryptedDatabaseSecretStore(BaseSecretStore):
    """AES-GCM encrypted local secret storage protected by a server master key."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        master_key: SecretStr | None,
    ) -> None:
        self._session_factory = session_factory
        self._key = _decode_master_key(master_key) if master_key is not None else None

    @property
    def available(self) -> bool:
        return self._key is not None

    def put(self, secret_id: str, value: SecretStr) -> None:
        if self._key is None:
            raise ConfigurationError("encrypted credential storage is not configured")
        normalized_id = _validate_secret_id(secret_id)
        plaintext = value.get_secret_value()
        if not plaintext.strip():
            raise ConfigurationError("provider credential cannot be empty")
        nonce = secure_random.token_bytes(12)
        ciphertext = AESGCM(self._key).encrypt(
            nonce,
            plaintext.encode("utf-8"),
            normalized_id.encode("utf-8"),
        )
        with session_scope(self._session_factory) as session:
            row = session.get(EncryptedSecretRecordModel, normalized_id)
            if row is None:
                session.add(
                    EncryptedSecretRecordModel(
                        secret_id=normalized_id,
                        ciphertext=ciphertext,
                        nonce=nonce,
                    )
                )
            else:
                row.ciphertext = ciphertext
                row.nonce = nonce
                row.updated_at = datetime.now(UTC)

    def resolve(self, secret_id: str) -> SecretStr:
        if self._key is None:
            raise ConfigurationError("encrypted credential storage is not configured")
        normalized_id = _validate_secret_id(secret_id)
        with session_scope(self._session_factory) as session:
            row = session.get(EncryptedSecretRecordModel, normalized_id)
            if row is None:
                raise ConfigurationError("stored credential is not configured")
            ciphertext = bytes(row.ciphertext)
            nonce = bytes(row.nonce)
        try:
            plaintext = AESGCM(self._key).decrypt(
                nonce,
                ciphertext,
                normalized_id.encode("utf-8"),
            )
        except InvalidTag as exc:
            raise ConfigurationError("stored credential failed integrity validation") from exc
        try:
            decoded = plaintext.decode("utf-8")
        except UnicodeDecodeError as exc:  # pragma: no cover - authenticated ciphertext guards this
            raise ConfigurationError("stored credential failed integrity validation") from exc
        if not decoded.strip():
            raise ConfigurationError("stored credential is not configured")
        return SecretStr(decoded)

    def contains(self, secret_id: str) -> bool:
        normalized_id = _validate_secret_id(secret_id)
        with session_scope(self._session_factory) as session:
            return session.get(EncryptedSecretRecordModel, normalized_id) is not None

    def delete(self, secret_id: str) -> bool:
        normalized_id = _validate_secret_id(secret_id)
        with session_scope(self._session_factory) as session:
            row = session.get(EncryptedSecretRecordModel, normalized_id)
            if row is None:
                return False
            session.delete(row)
            return True


class CompositeSecretResolver(BaseSecretResolver):
    """Resolve environment and encrypted-database references without exposing values."""

    def __init__(
        self,
        environment: EnvironmentSecretResolver,
        stored: BaseSecretStore | None,
    ) -> None:
        self._environment = environment
        self._stored = stored

    def resolve(self, credential_ref: str) -> SecretStr:
        if _ENV_REF.fullmatch(credential_ref):
            return self._environment.resolve(credential_ref)
        secret_id = stored_secret_id(credential_ref)
        if self._stored is None:
            raise ConfigurationError("encrypted credential storage is not configured")
        return self._stored.resolve(secret_id)

    def is_configured(self, credential_ref: str | None) -> bool:
        if credential_ref is None:
            return False
        if _ENV_REF.fullmatch(credential_ref):
            return self._environment.is_configured(credential_ref)
        secret_id = stored_secret_id(credential_ref)
        if self._stored is None or not self._stored.available:
            return False
        try:
            self._stored.resolve(secret_id)
        except ConfigurationError:
            return False
        return True


def new_provider_secret_reference(provider_id: str) -> tuple[str, str]:
    if re.fullmatch(r"[a-z0-9][a-z0-9._-]{1,99}", provider_id) is None:
        raise ConfigurationError("provider identity is invalid")
    secret_id = f"provider/{provider_id}/{secure_random.token_hex(6)}"
    return secret_id, f"secret:{secret_id}"


def stored_secret_id(credential_ref: str) -> str:
    match = _SECRET_REF.fullmatch(credential_ref)
    if match is None:
        raise ConfigurationError("credential reference scheme is not supported")
    return match.group(1)


def _validate_secret_id(secret_id: str) -> str:
    if _SECRET_REF.fullmatch(f"secret:{secret_id}") is None:
        raise ConfigurationError("stored credential identity is invalid")
    return secret_id


def _decode_master_key(master_key: SecretStr) -> bytes:
    encoded = master_key.get_secret_value().strip()
    try:
        padding = "=" * (-len(encoded) % 4)
        decoded = base64.b64decode(encoded + padding, altchars=b"-_", validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ConfigurationError(
            "MM_SECRET_MASTER_KEY must be URL-safe base64 for exactly 32 bytes"
        ) from exc
    if len(decoded) != 32:
        raise ConfigurationError(
            "MM_SECRET_MASTER_KEY must be URL-safe base64 for exactly 32 bytes"
        )
    return decoded

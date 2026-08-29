FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        fonts-noto-cjk \
        texlive-bibtex-extra \
        texlive-lang-chinese \
        texlive-latex-extra \
        texlive-xetex \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 65532 --no-create-home --shell /usr/sbin/nologin compiler

COPY sandbox/paper-entrypoint.sh /usr/local/bin/mathmodel-paper-compile
RUN chmod 0555 /usr/local/bin/mathmodel-paper-compile

USER 65532:65532
WORKDIR /output
ENTRYPOINT ["/usr/local/bin/mathmodel-paper-compile"]

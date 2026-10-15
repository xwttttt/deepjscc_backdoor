"""Inventory and screen files selected for a GitHub project upload."""

from __future__ import annotations

import collections
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "private_key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "github_token": re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})"),
    "aws_access_key": re.compile(rb"(?:AKIA|ASIA)[A-Z0-9]{16}"),
    "assigned_secret": re.compile(
        rb"(?im)^\s*(?:password|passwd|api_key|access_token|secret_key)\s*[:=]\s*[\"'][^\"'\r\n]{8,}[\"']"
    ),
}


def main() -> None:
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT,
    )
    paths = sorted(set(output.decode("utf-8").split("\0")) - {""})
    totals = collections.defaultdict(lambda: {"files": 0, "bytes": 0})
    findings = []
    files = []
    for relative in paths:
        path = ROOT / relative
        if not path.is_file():
            continue
        size = path.stat().st_size
        files.append({"path": relative, "bytes": size})
        totals[relative.split("/")[0]]["files"] += 1
        totals[relative.split("/")[0]]["bytes"] += size
        if size > 100 * 1024 * 1024:
            findings.append({"path": relative, "kind": "github_file_size_limit"})
        if path.name.lower() in {".env", "credentials", "id_rsa", "id_ed25519"} or path.suffix.lower() in {".pem", ".key", ".p12", ".pfx"}:
            findings.append({"path": relative, "kind": "credential_filename"})
        if size < 8 * 1024 * 1024:
            content = path.read_bytes()
            if b"\0" not in content[:4096]:
                for name, pattern in PATTERNS.items():
                    if pattern.search(content):
                        findings.append({"path": relative, "kind": name})
    report = {"files": files, "totals": dict(totals), "findings": findings}
    report_path = ROOT / "tmp/github_upload_audit.json"
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"file_count": len(files), "bytes": sum(x["bytes"] for x in files),
                      "totals": dict(totals), "findings": findings,
                      "report": str(report_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

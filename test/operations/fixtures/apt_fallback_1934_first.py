# Historical negative control: PR lukstafi/ocannl-staging#1934, first commit
# f74120cfd6654f43ba371312742f1862cad7c7ca, second Python heredoc verbatim.
from pathlib import Path
import re

host = "azure.archive.ubuntu.com"
found = False
for path in (Path("/etc/apt/sources.list.d/ubuntu.sources"), Path("/etc/apt/sources.list")):
    if not path.exists():
        continue
    source = path.read_text()
    if host not in source:
        continue
    found = True
    if path.suffix == ".sources":
        blocks = []
        for block in source.split("\n\n"):
            if host not in block:
                blocks.append(block)
                continue
            suites = re.search(r"^Suites: (.+)$", block, re.M)
            if not suites:
                raise SystemExit(f"::error::Cannot find Suites in {path}'s Azure source")
            names = suites[1].split()
            for security in (False, True):
                selected = [s for s in names if s.endswith("-security") == security]
                if selected:
                    part = re.sub(r"^Suites: .+$", "Suites: " + " ".join(selected), block, flags=re.M)
                    blocks.append(part.replace(host, "security.ubuntu.com" if security else "archive.ubuntu.com"))
        source = "\n\n".join(blocks)
    else:
        source = "\n".join(
            line.replace(host, "security.ubuntu.com" if re.search(r"\S+-security(?:\s|$)", line) else "archive.ubuntu.com")
            for line in source.split("\n")
        )
    path.write_text(source)
    print(f"Fallback Ubuntu sources in {path}:\n{source}", flush=True)
if not found:
    raise SystemExit("::error title=apt-get update::Cannot switch mirrors: neither Ubuntu source file contains azure.archive.ubuntu.com")

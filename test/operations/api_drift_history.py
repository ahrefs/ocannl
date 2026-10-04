"""Exercise the real API-drift CLI over a scratch first-parent merge history."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

reader = str(Path(sys.argv[1]).resolve())

with tempfile.TemporaryDirectory(prefix="946-api-drift-") as scratch:
    root = Path(scratch)
    env = dict(os.environ, GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
               GIT_AUTHOR_NAME="Fixture", GIT_AUTHOR_EMAIL="fixture@example.invalid",
               GIT_COMMITTER_NAME="Fixture", GIT_COMMITTER_EMAIL="fixture@example.invalid")

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, env=env, text=True).strip()

    def write(path, text):
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)

    def commit(subject):
        git("add", ".")
        git("commit", "-m", subject)
        return git("rev-parse", "HEAD")

    def read(since, until="HEAD", success=True):
        result = subprocess.run([reader, since, until], cwd=root, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert (result.returncode == 0) == success, result.stdout + result.stderr
        return result.stdout + result.stderr

    git("init", "-b", "master")
    write("arrayjit/lib/cap.mli", "type t = { old_scope : bool }\nval run :\n int -> int\n")
    write("lib/implicit.ml", "let public_value = 1\n")
    write("lib/hidden.ml", "let hidden = 1\n")
    write("lib/hidden.mli", "val public : int\n")
    write("bin/private.ml", "let private_value = 1\n")
    write("arrayjit/lib/dune", "(library (name backend) (public_name pkg.backend) (modules impl) (libraries (select impl.ml from (cuda -> impl.cuda.ml) (-> impl.missing.ml))))\n")
    write("arrayjit/lib/impl.mli", "val hidden : int\n")
    write("arrayjit/lib/impl.cuda.ml", "let hidden = 1\n")
    write("arrayjit/lib/impl.missing.ml", "let hidden = 1\n")
    write("tensor/dune", "(ocamllex lexer private_lexer)\n(menhir (modules parser))\n(library (name parserlib) (public_name pkg.parserlib) (modules lexer parser))\n(executable (name private) (modules private_lexer))\n")
    write("tensor/parser.mly", "%token OLD\n%%\n")
    write("tensor/lexer.mll", "{let exported = 1}\nrule token = parse | eof { () }\n")
    write("tensor/private_lexer.mll", "{let internal = 1}\nrule token = parse | eof { () }\n")
    base = commit("base")
    git("checkout", "-b", "feature")
    write("arrayjit/lib/cap.mli", "type t = { scopes : int list }\nval run :\n int -> string\n")
    write("lib/implicit.ml", 'let public_value = "new inferred type"\n')
    write("lib/hidden.ml", 'let hidden = "not an export"\n')
    write("bin/private.ml", 'let private_value = "also not public"\n')
    write("arrayjit/lib/impl.cuda.ml", "let hidden = true\n")
    write("arrayjit/lib/impl.missing.ml", "let hidden = true\n")
    write("tensor/parser.mly", "%token NEW\n%%\n")
    write("tensor/lexer.mll", "{let exported = true}\nrule token = parse | eof { () }\n")
    write("tensor/private_lexer.mll", "{let internal = true}\nrule token = parse | eof { () }\n")
    side = commit("feature implementation")
    git("checkout", "master")
    git("merge", "--no-ff", "feature", "-m", "Merge pull request #624 from fixture/feature")
    merged = git("rev-parse", "HEAD")
    report = read(base)
    assert f"commit {merged} Merge pull request #624" in report
    assert f"commit {side}" not in report
    assert "old_scope" in report and "scopes" in report and "int -> string" in report
    assert "public_value" in report and "new inferred type" in report
    assert "lib/hidden.ml" not in report and "bin/private.ml" not in report
    assert "impl.cuda.ml" not in report and "impl.missing.ml" not in report
    assert "tensor/parser.mly" in report and "%token NEW" in report
    assert "tensor/lexer.mll" in report and "exported = true" in report
    assert "private_lexer" not in report
    print("merge attribution, multiline signatures, implicit exports and exclusions: pass")
    print("select interfaces exclude arm bodies; public generator input changes are visible: pass")

    # A change that later reverts must remain attributed to both introducing commits.
    write("arrayjit/lib/cap.mli", "type t = { old_scope : bool }\nval run : int -> int\n")
    reverted = commit("Revert API change (#625)")
    report = read(base)
    assert f"commit {merged}" in report and f"commit {reverted}" in report
    print("reverted declarations retain both first-parent attribution points: pass")

    write("lib/implicit.mli", "val public_value : string\n")
    narrowed = commit("Publish an explicit interface (#626)")
    report = read(reverted)
    assert "- let public_value" in report and "+ val public_value" in report
    assert f"commit {narrowed}" in report
    print("adding an interface records retirement of the implicit source surface: pass")

    git("rm", "lib/implicit.mli")
    removed = commit("Remove an interface (#627)")
    report = read(narrowed)
    assert "- val public_value" in report and "+ let public_value" in report
    print("removing an interface exposes the implementation surface: pass")

    write("arrayjit/lib/cap.mli", "(** prose only *)\ntype t = { old_scope : bool }\nval run :\n int -> int\n")
    documented = commit("Document API (#628)")
    assert "0 declaration changes" in read(removed, documented)
    write("arrayjit/lib/cap.mli", '[@@@ocaml.text "floating docs"]\ntype t = { old_scope : bool }\nval run : int -> int\n')
    floating = commit("Floating documentation only")
    assert "0 declaration changes" in read(documented, floating)
    assert "first-parent history" in read(side, success=False)
    assert "git failed" in read("missing-revision", success=False)
    assert "0 declaration changes across 0" in read(base, base)
    print("documentation-only edits, empty windows and invalid endpoints: pass")

    git("rm", "arrayjit/lib/impl.mli")
    exposed = commit("Expose selected implementation")
    report = read(floating, exposed)
    assert "impl.cuda.ml" in report and "impl.missing.ml" in report
    assert "hidden = true" in report
    print("select arms become visible when the compiled target interface is removed: pass")
    write("lib/initializer.ml", 'let value = 1\nlet () = print_endline "old";;\nprint_endline "old bare";;\n')
    initialized = commit("Initial module with unnamed initialization")
    write("lib/initializer.ml", 'let value = 1\nlet () = print_endline "new";;\nprint_endline "new bare";;\n')
    quiet = commit("Change non-exporting initialization")
    assert "0 declaration changes" in read(initialized, quiet)
    print("non-exporting initializer and evaluation edits do not create API entries: pass")
    write("arrayjit/lib/cap.mli", "val")
    commit("Invalid source must refuse")
    assert "api-drift:" in read(quiet, success=False)
    print("invalid source refuses the real historical reader: pass")

(** The agent guide stays within the size the harness will actually load.

    [AGENTS.md] is imported into every coding session through [CLAUDE.md], and Claude Code truncates
    an imported instructions file at 32 KiB -- silently, from the end, which is where the Pull
    Requests, Configuration and Syntax sections sit. The file crossed that line one review-fix
    bullet at a time, each one a rule with its whole mechanism and failure story attached, and
    nothing said so until a session was found working from a guide whose tail it had never seen. The
    trim that brought it back (lukstafi/ocannl-staging, September 2026) moved the mechanisms into
    [docs/agent-notes/] and left the rules with pointers; this scan is what keeps the next bullet
    from undoing it.

    The decision is the byte count of the file dune hands over, against the cap. Nothing about the
    content is read, so the scan cannot go stale on a rewrite, and the count itself goes to stderr:
    a size in the golden would move on every edit and turn each one into a promote (gh-ocannl-665).
    A floor keeps the claim from passing vacuously on an empty or misrouted file, and a synthetic
    control holds the refusal against a byte string built to cross the cap, since the live file --
    on a good day -- never does. *)

open Base
open Stdio
open Verdict.Claims

(** Claude Code's cap on one imported instructions file, in bytes. *)
let cap = 32 * 1024

(** A guide this small has lost most of its sections: the trimmed file is around 29 KiB, and the
    orientation alone (structure, build commands, key concepts) is well past 8 KiB. *)
let floor = 8 * 1024

(** The one decision, shared by the live claim and its control: whether [bytes] fits under the cap.
    Strictly under, so a file AT the cap is refused too -- the harness's boundary behavior is not
    documented, and a margin of one byte is not a margin. *)
let fits ~bytes = bytes < cap

(** The warning band's lower edge: about a kilobyte below the cap, which is a few bullets of
    headroom -- enough to trim deliberately rather than under a red build. A guide in the band still
    passes (the band is advice, not a second cap), and the warning goes to stderr with the count, so
    the golden does not move when the file enters or leaves the band. *)
let warn_at = 31 * 1024

(** Whether [bytes] sits in the warning band: under the cap, at or above [warn_at]. *)
let warns ~bytes = fits ~bytes && bytes >= warn_at

let () =
  let path =
    match Array.to_list Stdlib.Sys.argv |> List.tl_exn with
    | [ path ] -> path
    | args ->
        eprintf "FAILED: expected exactly one path, the agent guide, got %d arguments\n"
          (List.length args);
        Stdlib.exit 1
  in
  let content = Stdlib.In_channel.with_open_bin path Stdlib.In_channel.input_all in
  let bytes = String.length content in
  eprintf "AGENTS.md: %d bytes; cap %d, warning band from %d, floor %d (not part of the golden)\n"
    bytes cap warn_at floor;
  if warns ~bytes then
    eprintf
      "WARNING: AGENTS.md is %d bytes, within %d bytes of the 32 KiB cap; trim it before the next \
       rule lands (docs/agent-notes/build-and-test.md#repository-scans, the AGENTS.md bullet)\n"
      bytes (cap - bytes);
  (* The guide's identity, held independently of its size: the file dune handed over is named
     AGENTS.md and opens with the guide's own title line. A floor alone accepts any prose of the
     right size, and a misrouted rule would then measure the wrong document forever. *)
  let is_the_guide =
    String.equal (Stdlib.Filename.basename path) "AGENTS.md"
    && String.is_prefix content ~prefix:"# OCANNL Agent Guide\n"
  in
  printf
    "Size of AGENTS.md against the cap Claude Code applies to an imported instructions file. The\n\
     byte count goes to stderr, since a tally in a golden moves on every edit (gh-ocannl-665).\n\n";
  p "the scan was handed the agent guide, not an empty or misrouted file"
    (is_the_guide && bytes >= floor);
  p "AGENTS.md is under 32 KiB, so no session works from a truncated guide" (fits ~bytes);
  (* The control: the refusal on a byte count built to cross the cap, and the acceptance one byte
     short of it. The inputs are LITERALS, not derived from [cap] -- a control fed [cap] itself
     holds for any positive cap, so a cap edited to 64 KiB would pass it and the live file would
     grow into truncation unnoticed. Written this way, a cap that drifts or a comparison that flips
     fails here first. *)
  p "a synthesized guide of exactly 32 KiB is refused" (not (fits ~bytes:32_768));
  p "a synthesized guide one byte under the cap is accepted" (fits ~bytes:32_767);
  (* The warning band's controls, literal for the same reason: its lower edge warns, one byte below
     it does not, and a guide at the cap is refused rather than warned about. *)
  p "a synthesized guide at the warning band's lower edge is warned about" (warns ~bytes:31_744);
  p "a synthesized guide one byte below the warning band is not warned about"
    (not (warns ~bytes:31_743));
  p "a synthesized guide of exactly 32 KiB is refused, not warned about" (not (warns ~bytes:32_768));
  Test_scan.Refusal_control_manifest.print "agents_md_size.ml"

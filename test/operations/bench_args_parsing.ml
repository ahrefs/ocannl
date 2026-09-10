(* gh-ocannl-634: the commandline split the bin/ tools share ([Bench_args]).

   The defect this module exists to prevent is silent: a negative extent filtered away as if it were
   an option does not stop the run, it shifts every later positional one slot left, and the bench
   then measures a geometry nobody asked for and reports a plausible number for it. Two copies of
   the idiom had that bug, and the two that were fixed carried the predicate twice, verbatim -- so
   what is pinned here is the predicate itself, the slot alignment it protects, and the range check
   that now happens where each argument is read.

   The tool is driven through [?argv] rather than by running a bench: this is a parsing test, and a
   real bench would need a backend, a compilation and minutes of measurement to say the same thing.

   One line of stderr in a passing run is deliberate: the shadowing check below hands [Bench_args] a
   post-[--] argument that spells a configuration key, and the warning that draws is the thing being
   tested. *)

open Base
open Stdio
open Verdict.Claims

(* The five positionals of `schedule_bench 256 20 -64 512`, the case from the issue, with the
   library's own flags interleaved the way a real invocation carries them. *)
let flagged = [| "schedule_bench"; "--ocannl_backend=cc"; "256"; "20"; "-64"; "512" |]

(* [invalid_arg]'s message, or [None] when the thunk returned. *)
let refused f = match f () with _ -> None | exception Invalid_argument msg -> Some msg

let () =
  let args = Bench_args.create ~argv:flagged "schedule_bench" in
  (* The negative stays in its own slot: n, repeats, m, k -- not n, repeats, k shifted into m. *)
  p "a negative positional keeps every later argument in its slot"
    (List.equal String.equal (Bench_args.positional args) [ "256"; "20"; "-64"; "512" ]);
  p "the trailing argument is still read from slot 3"
    (Bench_args.int args 3 ~name:"k" ~default:0 = 512);
  (* And the negative itself is refused by name, rather than measured or lost. *)
  p "a negative extent is refused, naming the tool, the argument and the value"
    (match refused (fun () -> Bench_args.int args 2 ~name:"m" ~default:0) with
    | Some msg ->
        String.is_prefix msg ~prefix:"schedule_bench: "
        && String.is_substring msg ~substring:"m must be positive, got -64"
    | None -> false)

let () =
  (* Every argument form the tools use, against the one predicate. A [-] followed by a non-digit is
     an option (a short flag); a [-] followed by a digit is a value; a lone [-] is neither, so it
     stays a positional rather than disappearing. *)
  let cases =
    [
      ("--ocannl_backend=cc", true);
      ("--bm=64", true);
      ("--", true) (* ends the options; not a positional itself *);
      ("-v", true);
      ("-64", false);
      ("-0", false);
      ("-", false);
      ("512", false);
      ("f32", false);
      ("guard", false);
    ]
  in
  List.iter cases ~f:(fun (s, expected) ->
      printf "%-22s option: %b\n" s (Bench_args.is_option s);
      Verdict.claim (s ^ " classified as documented") (Bool.equal (Bench_args.is_option s) expected))

let () =
  (* A lone [--] is the escape for an argument that must be taken literally: everything after it is
     positional even when it looks exactly like a flag. This is the only way to hand [gpt2_generate]
     a prompt beginning with a dash. *)
  let argv = [| "gpt2_generate"; "--ocannl_backend=cc"; "--"; "--why the long face"; "5" |] in
  let args = Bench_args.create ~argv "gpt2_generate" in
  p "after a lone [--] a flag-looking argument is the positional"
    (String.equal (Bench_args.string args 0 ~default:"") "--why the long face");
  p "positionals after [--] keep their slots"
    (Bench_args.int args 1 ~name:"num_tokens" ~default:20 = 5);
  (* The escape governs this split only: the library reads its own settings in a separate pass over
     the whole argv, with no terminator and prefix-free spellings accepted, so a post-[--] argument
     that spells a setting has already been applied by the time a tool sees it. Unfixable from here
     -- what is fixable is the silence, so the collision is reported. *)
  p_empty "an ordinary post-[--] argument shadows no setting" ~over:(Bench_args.positional args)
    (Bench_args.shadowing_config args);
  let shadowing =
    Bench_args.create ~argv:[| "gpt2_generate"; "--"; "--backend=cuda" |] "gpt2_generate"
  in
  p "a post-[--] argument that spells a setting is reported as shadowing it"
    (List.equal String.equal (Bench_args.shadowing_config shadowing) [ "--backend=cuda" ]);
  p "and is still this tool's positional"
    (String.equal (Bench_args.string shadowing 0 ~default:"") "--backend=cuda")

let () =
  let args = Bench_args.create ~argv:[| "narrow_gebp_bench"; "bf16"; "512" |] "narrow_gebp_bench" in
  p "a word positional is read as itself"
    (String.equal (Bench_args.string args 0 ~default:"f32") "bf16");
  p "a missing positional falls back to the default"
    (Bench_args.int args 2 ~name:"repeats" ~default:20 = 20);
  (* [--bm=]/[--bk=] beat the positional they duplicate, and a repeated flag resolves the way the
     library resolves a repeated setting: [Utils.read_cmdline_var] is an [Array.find_map] over argv,
     so the FIRST spelling wins, here as there. *)
  let args =
    Bench_args.create
      ~argv:[| "narrow_gebp_bench"; "f32"; "512"; "20"; "32"; "--bm=16"; "--bm=64" |]
      "narrow_gebp_bench"
  in
  p "a --flag= overrides the positional it duplicates, first spelling winning as in the library"
    (Bench_args.int args 3 ~flag:"bm" ~name:"bm" ~default:64 = 16);
  p "without the flag the same slot is still read"
    (Bench_args.int
       (Bench_args.create
          ~argv:[| "narrow_gebp_bench"; "f32"; "512"; "20"; "32" |]
          "narrow_gebp_bench")
       3 ~flag:"bm" ~name:"bm" ~default:64
    = 32)

let () =
  let args = Bench_args.create ~argv:[| "narrow_storage_bench"; "1024"; "50"; "0" |] "b" in
  (* [least:0] is for the counts whose zero is documented -- narrow_storage_bench's whole-pool
     [threads], schedule_bench's skipped naive leg. The default domain rejects the same value. *)
  p "a documented zero passes under least:0"
    (Bench_args.int args 2 ~name:"threads" ~least:0 ~default:1 = 0);
  p "the same zero is refused as an extent"
    (match refused (fun () -> Bench_args.int args 2 ~name:"n" ~default:1) with
    | Some msg -> String.is_substring msg ~substring:"n must be positive, got 0"
    | None -> false);
  p "a non-integer positional is refused by name"
    (match
       refused (fun () ->
           Bench_args.int (Bench_args.create ~argv:[| "b"; "wide" |] "b") 0 ~name:"n" ~default:1)
     with
    | Some msg -> String.is_substring msg ~substring:"n must be an integer, got \"wide\""
    | None -> false);
  (* The default is checked against the same domain: a tool cannot quietly default out of it. *)
  p "a default outside the domain is refused too"
    (match
       refused (fun () ->
           Bench_args.int (Bench_args.create ~argv:[| "b" |] "b") 0 ~name:"n" ~default:0)
     with
    | Some msg -> String.is_substring msg ~substring:"n must be positive, got 0"
    | None -> false)

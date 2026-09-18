(** Positional arguments beside the [--ocannl_*] configuration flags (gh-ocannl-634).

    Every tool here takes its geometry positionally -- [schedule_bench 512 20 256] -- while the
    library's own settings arrive on the same commandline, as [--ocannl_*] flags. Splitting the two
    is a three-line idiom, and it had been written five times, twice wrongly: filtering out
    everything that starts with a [-] drops a negative extent along with the flags, and where there
    are several positionals it does not merely lose that value, it shifts every later argument one
    slot left. The bench then runs at a geometry nobody asked for and reports a plausible number for
    it, which is the failure mode worth spending a module on.

    So the predicate lives here once: an option is [--]-prefixed, or a [-] followed by a non-digit;
    a bare [-64] is a positional and reaches the range check below, which names it. A lone [--] ends
    the options, so an argument that must be taken literally has an escape (a [gpt2_generate] prompt
    beginning with a dash, say) -- the filter is deliberately blind to what a tool expects in each
    slot, and cannot tell a flag from a prompt.

    That escape governs THIS split only. The library reads its own settings in a separate pass over
    the whole of [Sys.argv] at startup ([Utils.read_cmdline_var]), which knows no terminator and
    accepts prefix-free spellings, so a post-[--] argument that happens to spell a known setting
    ([-- --backend=cuda] as a prompt) has already been applied as configuration before any of this
    runs, and nothing here can unset it. What this module can do is refuse to let that happen
    quietly: every post-terminator positional is checked against {!Utils.cmdline_arg_is_config_key}
    and reported on stderr, so the collision is loud rather than a run at a configuration nobody
    chose. See {!shadowing_config}.

    Parsing and validation are one call, not two: an extent that arrives as [0] or [-64] is rejected
    where it is read, by name, rather than reaching an [Array.init] or a schedule somewhere below.
*)

open Base

type t = {
  tool : string;  (** Prefixes every diagnostic, so a failure names the tool that failed. *)
  positional : string list;
  options : string list;  (** Kept for [int]'s [?flag]: the [--name=value] arguments, in order. *)
  shadowing : string list;
      (** Post-[--] positionals the library's own commandline scan also claims. *)
}

(** An option is [--]-prefixed (the library's [--ocannl_*] settings, and any of a tool's own), or a
    [-] followed by a non-digit. Everything else -- including [-64] and a bare [-] -- is positional.
*)
let is_option s =
  String.is_prefix s ~prefix:"--"
  || (String.length s > 1 && Char.equal s.[0] '-' && not (Char.is_digit s.[1]))

(** [create tool] splits [Sys.get_argv ()] -- or [?argv], which is how the test drives it -- into
    positionals and options. [argv.(0)] is the program name and is neither. *)
let create ?argv tool =
  let argv = match argv with Some a -> a | None -> Sys.get_argv () in
  let rest = match Array.to_list argv with [] -> [] | _ :: rest -> rest in
  (* Before a lone [--] the predicate decides; after it, every argument is positional verbatim. *)
  let before, after =
    match List.split_while rest ~f:(fun s -> not (String.equal s "--")) with
    | before, _ :: after -> (before, after)
    | before, [] -> (before, [])
  in
  (* Only a post-terminator positional can look like a setting: before the terminator anything that
     does is an option, and never reaches a slot. *)
  let shadowing = List.filter after ~f:Utils.cmdline_arg_is_config_key in
  List.iter shadowing ~f:(fun arg ->
      Stdio.eprintf
        "%s warning: the positional %S also spells an OCANNL setting, which the library has \
         already applied from the commandline; [--] keeps it as this tool's argument but cannot \
         unset it.\n\
         %!"
        tool arg);
  {
    tool;
    positional = List.filter before ~f:(Fn.non is_option) @ after;
    options = List.filter before ~f:is_option;
    shadowing;
  }

let bad t fmt = Printf.ksprintf (fun msg -> invalid_arg (t.tool ^ ": " ^ msg)) fmt

(** The positional arguments, in order: for a tool whose arguments are words rather than extents (a
    mode, a subcommand). *)
let positional t = t.positional

(** [flag t ~name] is whether the bare option [--name] was passed — a tool's own switch, alongside
    [int]'s [--name=value] spelling. Exact match, so [--name=1] is not it: a switch that also
    accepted a value would silently ignore the value. *)
let flag t ~name = List.mem t.options ("--" ^ name) ~equal:String.equal

(** The post-[--] positionals that are also a spelling of a known configuration key, warned about at
    {!create}: the escape holds for this tool's slots, and the library has read them anyway. *)
let shadowing_config t = t.shadowing

(** [flag_value t ~flag] is the value of a [--flag=value] option, unparsed, or [None] when the tool
    was not given one -- for an argument that is not a single integer, such as the register-tile
    geometry [--tile=rm,rn,lanes] ([Bench_tile]). FIRST spelling wins, as in {!int}'s [?flag] and in
    the library's own scan ([Utils.read_cmdline_var]): one commandline should not have two
    precedence rules. *)
let flag_value t ~flag =
  let prefix = "--" ^ flag ^ "=" in
  List.find_map t.options ~f:(String.chop_prefix ~prefix)

(** [string t i ~default] is positional [i], or [default] when it was not given. *)
let string t i ~default = Option.value (List.nth t.positional i) ~default

let check t ~name ~least ~where v =
  if v >= least then v
  else
    bad t "%s must be %s, got %d%s" name
      (if least = 1 then "positive"
       else if least = 0 then "nonnegative"
       else Printf.sprintf "at least %d" least)
      v where

let parse t ~name ~least ~where s =
  match Option.try_with (fun () -> Int.of_string s) with
  | None -> bad t "%s must be an integer, got %S%s" name s where
  | Some v -> check t ~name ~least ~where v

(** [int t i ~name ~default] is positional [i] as an integer, validated on the spot: each of these
    arguments is an extent or a repeat count, so the domain is [>= least] -- 1 by default, and
    [~least:0] for the counts whose zero is documented, such as a leg the tool may skip.

    [?flag] names a [--flag=value] spelling that takes precedence over the positional, for an
    argument a tool wants reachable without counting slots. The default is checked too, so a tool
    cannot quietly default outside its own domain. *)
let int ?(least = 1) ?flag t i ~name ~default =
  let by_flag =
    Option.bind flag ~f:(fun flag ->
        let where = Printf.sprintf " (from --%s=)" flag in
        Option.map (flag_value t ~flag) ~f:(parse t ~name ~least ~where))
  in
  match by_flag with
  | Some v -> v
  | None -> (
      match List.nth t.positional i with
      | Some s -> parse t ~name ~least ~where:"" s
      | None -> check t ~name ~least ~where:" (the tool's default)" default)

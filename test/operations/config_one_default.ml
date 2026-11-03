(* gh-ocannl-1861: one key, one default, held at runtime. [Utils.get_global_arg] refuses a read of a
   key under a default other than the one this process first read it under, whatever the key's
   source -- the static check 9 of test_config_consistency covers only the library roots it scans,
   and a configuration that sets the key hides a second default without removing it. The rule runs
   this beside this directory's [ocannl_config], which sets [backend]; [default_prng_variant] is
   first read by [Utils.restore_settings] at module initialization, and [no_config_file] by the
   bootstrap. *)

open Base
open Verdict.Claims

(* The refusal [read] raises, if it raises one. *)
let refusal read =
  match read () with _ -> None | exception Invalid_argument message -> Some message

let names_both ~key ~earlier ~later message =
  List.for_all
    [ "ocannl_" ^ key; Printf.sprintf "default %S" later; Printf.sprintf "under %S" earlier ]
    ~f:(fun part -> String.is_substring message ~substring:part)

let records_of key =
  List.count (Utils.resolved_config ()) ~f:(fun r -> String.equal r.Utils.key key)

let () =
  let backend () = Utils.get_global_arg_with_source ~arg_name:"backend" ~default:"" in
  let first, source = backend () in
  let second, _ = backend () in
  p "a key read twice under one default resolves both times to one value"
    (String.equal first second);
  let records = records_of "backend" in
  let refused = refusal (fun () -> Utils.get_global_arg ~arg_name:"backend" ~default:"cc") in
  Option.iter refused ~f:(Stdio.eprintf "refused: %s\n");
  p "a second default is refused, naming both, though the config file sets the key"
    ((match source with Utils.From_default -> false | _ -> true)
    && Option.exists refused ~f:(names_both ~key:"backend" ~earlier:"" ~later:"cc"));
  p "a refused read records nothing, and the first default still reads"
    (records_of "backend" = records && String.equal first (fst (backend ())));
  p "a second default is refused for a key the library read first, at module initialization"
    (Option.exists
       (refusal (fun () -> Utils.get_global_arg ~arg_name:"default_prng_variant" ~default:"heavy"))
       ~f:(names_both ~key:"default_prng_variant" ~earlier:"light" ~later:"heavy"));
  p "a second default is refused for a key the bootstrap resolved"
    (Option.exists
       (refusal (fun () -> Utils.get_global_flag ~arg_name:"no_config_file" ~default:true))
       ~f:(names_both ~key:"no_config_file" ~earlier:"false" ~later:"true"))

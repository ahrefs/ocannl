(* gh-ocannl-2354: a key a test states in process is the value it reads, whatever the caller
   exported.

   The config-file table ([Utils.config_file_args]) ranks below the environment, so a test that
   wrote its setting there ran under the caller's [OCANNL_<KEY>] whenever one was exported -- the
   recipe AGENTS.md gives for a one-off configuration change -- and nothing said so. A statement
   through [Utils.state_config] ranks above every source a run is given; a commandline argument that
   contradicts it is refused at the statement.

   The rule runs this with [--ocannl_profile=reproducible] and [--ocannl_cpu_schedule_min_work=5] on
   its commandline. The hostile ambient variables are exported in process ([Unix.putenv]), which is
   what the environment level reads ([Sys.getenv]), so no rule's environment is needed and an
   ambient value of the caller's for these keys is overwritten first. Reads go through
   [Utils.profile_key_source], the full resolution without the read-site bookkeeping. *)

open Base
open Verdict.Claims

(* Registered keys, so every spelling of them is one the config usage scan knows; no backend is
   linked, so nothing reads them but this test. The probe is the key cpu_scratch_matmul_grid and
   three other tests state, the commandline key the one the rule's commandline states. *)
let probe_key = "cc_grid_min_updates"
let cmdline_key = "cpu_schedule_min_work"

(* A key the reproducible profile's payload states: [false]. *)
let profile_key = "cuda_fast_math"
let resolved key = Utils.profile_key_source key

let () =
  Unix.putenv (Utils.env_var_name probe_key) "1000000000";
  p "with no statement, the exported variable is what the probe key resolves to"
    (match resolved probe_key with
    | Some ("1000000000", Utils.From_env var) -> String.equal var (Utils.env_var_name probe_key)
    | _ -> false);
  (* The hazard the setter exists for, kept as the negative control: a raw write to the config-file
     table is silently outranked by the ambient variable. *)
  Hashtbl.set Utils.config_file_args ~key:probe_key ~data:"7";
  p "a raw write to the config-file table is outranked by the exported variable"
    (match resolved probe_key with Some ("1000000000", Utils.From_env _) -> true | _ -> false);
  Hashtbl.remove Utils.config_file_args probe_key;
  Utils.state_config ~key:probe_key ~data:"0";
  p "a stated key outranks the exported variable, and says it was stated in process"
    (match resolved probe_key with Some ("0", Utils.From_in_process) -> true | _ -> false);
  p "Utils.stated_config reads the statement back"
    (Option.equal String.equal (Utils.stated_config probe_key) (Some "0"));
  Utils.unstate_config probe_key;
  p "withdrawing the statement leaves the exported variable answering again"
    (match resolved probe_key with Some ("1000000000", Utils.From_env _) -> true | _ -> false)

(* A statement also outranks a profile picked on the commandline, the aggregate of the most
   immediate source a run is given. *)
let () =
  p "unstated, the profile key resolves to the commandline-picked profile's payload"
    (match resolved profile_key with
    | Some ("false", Utils.From_profile (Utils.Cmdline_level, "reproducible")) -> true
    | _ -> false);
  Unix.putenv (Utils.env_var_name profile_key) "false";
  Utils.with_stated_config
    [ (profile_key, "true") ]
    (fun () ->
      p "a stated key outranks a commandline-picked profile and an exported variable alike"
        (match resolved profile_key with
        | Some ("true", Utils.From_in_process) -> true
        | _ -> false))

(* [with_stated_config] restores the prior statement, or its absence, even when its body raises. *)
let () =
  Utils.state_config ~key:probe_key ~data:"11";
  (try
     Utils.with_stated_config
       [ (probe_key, "13"); (profile_key, "true") ]
       (fun () ->
         p "inside with_stated_config, its statements are read"
           (match (resolved probe_key, resolved profile_key) with
           | Some ("13", Utils.From_in_process), Some ("true", Utils.From_in_process) -> true
           | _ -> false);
         failwith "planted")
   with Failure _ -> ());
  p "after a raising body, the prior statement and the prior absence are back"
    (Option.equal String.equal (Utils.stated_config probe_key) (Some "11")
    && Option.is_none (Utils.stated_config profile_key));
  Utils.unstate_config probe_key

(* A statement's record is rendered as its own source and layer, with the setter as its origin. *)
let () =
  let rendered =
    Utils.with_stated_config
      [ (probe_key, "17") ]
      (fun () ->
        Option.map (resolved probe_key) ~f:(fun (value, source) ->
            Utils.config_record_to_json { Utils.key = probe_key; value; source }))
  in
  p "a stated key's record renders as an in-process source, layer and origin"
    (Option.equal String.equal rendered
       (Some
          {|{"ocannl_config":1,"key":"cc_grid_min_updates","value":"17","source":"in_process","layer":"in_process","profile":null,"origin":"Utils.state_config"}|}))

(* What a statement refuses, naming the key: a commandline argument it contradicts (the one source
   it cannot silently outrank), and a key the library resolves before any statement can be made. *)
let () =
  let refusal ~key ~data =
    match Utils.state_config ~key ~data with
    | () ->
        Utils.unstate_config key;
        None
    | exception Invalid_argument message -> Some message
  in
  p "a statement agreeing with the commandline is accepted"
    (Option.is_none (refusal ~key:cmdline_key ~data:"5"));
  p "a statement contradicting the commandline is refused, naming the key and the argument"
    (match refusal ~key:cmdline_key ~data:"6" with
    | Some message ->
        String.is_substring message ~substring:"~key:\"cpu_schedule_min_work\""
        && String.is_substring message ~substring:"--ocannl_cpu_schedule_min_work=5"
    | None -> false);
  p "a contradicted statement leaves no statement behind"
    (Option.is_none (Utils.stated_config cmdline_key));
  p_all ~min:4
    "a statement of a key in a spelling no read looks up, or one no read registers, is refused, \
     naming the key"
    [
      "CC_GRID_MIN_UPDATES";
      "ocannl_cc_grid_min_updates";
      "cc-grid-min-updates";
      "cc_grid_min_update";
    ] ~f:(fun key ->
      match refusal ~key ~data:"0" with
      | Some message -> String.is_substring message ~substring:(Printf.sprintf "~key:%S" key)
      | None -> false);
  p_all ~min:2 "a statement of a key resolved at initialization is refused, naming the key"
    [ "profile"; "log_config_json" ] ~f:(fun key ->
      match refusal ~key ~data:"true" with
      | Some message -> String.is_substring message ~substring:(Printf.sprintf "~key:%S" key)
      | None -> false);
  let key = "print_decimals_precision" in
  let before = Utils.settings.print_decimals_precision in
  let data = Int.to_string (if Int.equal before 7 then 8 else 7) in
  p "a statement of a settings snapshot is refused, naming the key"
    (match Utils.state_config ~key ~data with
    | () ->
        Stdlib.Printf.eprintf "statement accepted: lookup=%b, snapshot still %d (before %d)\n"
          (Option.equal String.equal (Utils.stated_config key) (Some data))
          Utils.settings.print_decimals_precision before;
        Utils.unstate_config key;
        false
    | exception Invalid_argument message ->
        String.is_substring message ~substring:(Printf.sprintf "~key:%S" key))

let () =
  p_all ~min:3 "a startup-only input or compatibility spelling is refused, naming the key"
    [ ("log_level", "2"); ("detach_c_stderr", "false"); ("big_models", "true") ]
    ~f:(fun (key, data) ->
      match Utils.state_config ~key ~data with
      | () ->
          Utils.unstate_config key;
          false
      | exception Invalid_argument message ->
          String.is_substring message ~substring:(Printf.sprintf "~key:%S" key)
          && String.is_substring message ~substring:"resolved while the library initializes"
          && Option.is_none (Utils.stated_config key))

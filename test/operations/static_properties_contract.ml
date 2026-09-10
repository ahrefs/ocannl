(* gh-ocannl-710: the [Backend_intf.static_properties] contract, pinned generically.

   The four backends implementing [static_properties] used to agree on nothing but the outermost
   atom: CUDA and HIP emitted [Sexp.message]-shaped [(device (k v) ...)] entries, Metal and cc
   wrapped the same pairs one nesting level deeper, and Multidev emitted no device entries at all --
   a group atom followed by [(device_name CPU) (num_devices 16)], which a generic reader
   ([bin/device_props], the first caller these functions ever had) duly reported as two devices,
   neither of which existed, on the one backend whose entire purpose is multi-device debugging.

   The contract is stated once, on [Backend_intf.parse_static_properties], and read once, by that
   function; this test checks it against each backend's REAL dump, plus the parts of it a parser
   cannot enforce (uniform keys, the ordinal sequence).

   Two legs run in every configuration: the selected backend (so pinning [OCANNL_BACKEND] covers cc,
   metal, cuda and hip in turn -- the goldens stay backend-uniform because every claim is
   shape-only) and multidev_cc, which is always available and is the backend the issue was about.
   Device names, counts and attribute values are machine-specific, so they go to stderr; stdout
   carries claims only.

   The three negative controls are what make the leg claims non-vacuous: they are the shapes this
   change removed, and a parser that accepted them would let both legs pass on the old dumps. *)

open Base
open Ocannl
open Operation.DSL_modules
module BI = Ir.Backend_intf

let pair key value = Sexp.List [ Sexp.Atom key; value ]

(* The pre-gh-710 Multidev dump: a group atom followed by ordinary backend-level pairs. *)
let legacy_multidev_dump =
  Sexp.List
    [
      Sexp.Atom "multidev_cc_devices";
      pair "device_name" (Sexp.Atom "CPU");
      pair "num_devices" (Sexp.Atom "16");
    ]

(* The pre-gh-710 Metal / cc dump: one device, but its pairs wrapped in a list of their own. *)
let legacy_nested_dump =
  Sexp.List
    [
      Sexp.Atom "metal_devices";
      Sexp.List
        [
          Sexp.Atom "device";
          Sexp.List [ pair "device_name" (Sexp.Atom "GPU"); pair "device_ordinal" (Sexp.Atom "0") ];
        ];
    ]

(* An unlinked backend's report: deliberately not a device dump, and says so in its group atom. *)
let missing_backend_dump =
  Sexp.List [ Sexp.Atom "cuda_missing"; pair "error" (Sexp.Atom "Backend cuda missing") ]

let check_negative_controls () =
  Verdict.p "a dump whose children are backend-level pairs is not read as devices"
    (Option.is_none (BI.parse_static_properties legacy_multidev_dump));
  Verdict.p "a dump whose device pairs are one nesting level deeper is not read as devices"
    (Option.is_none (BI.parse_static_properties legacy_nested_dump));
  Verdict.p "a dump whose group atom does not name devices is not read as devices"
    (Option.is_none (BI.parse_static_properties missing_backend_dump))

let check_dump ~leg ~backend_name (props : Sexp.t) =
  Stdio.eprintf "%s (%s) static_properties:\n%s\n%!" leg backend_name (Sexp.to_string_hum props);
  (* Read off the top-level atom directly rather than from the parse, so that a dump failing the
     contract for some other reason still reports this claim on its own merits. *)
  Verdict.pf "%s: group atom is the backend name plus _devices" leg
    (match props with
    | Sexp.List (Sexp.Atom group :: _) -> String.equal group (backend_name ^ "_devices")
    | _ -> false);
  let parsed = BI.parse_static_properties props in
  Verdict.pf "%s: dump is a device dump (message-shaped device entries under the group atom)" leg
    (Option.is_some parsed);
  (* Every claim below is conjoined with "there is a device to check", so a dump that does not parse
     -- or enumerates nothing -- reports them as false rather than vacuously true: an unevaluated
     claim printing the same line as a verified one is what a golden cannot show. *)
  let devices = match parsed with Some { BI.devices; _ } -> devices | None -> [] in
  let some_device = not (List.is_empty devices) in
  Verdict.pf "%s: at least one device is enumerated" leg some_device;
  let keys = List.map devices ~f:(fun fields -> List.map fields ~f:fst) in
  Verdict.p_all (Printf.sprintf "%s: keys within a device entry are distinct" leg) keys
    ~f:(fun ks -> List.length (List.dedup_and_sort ks ~compare:String.compare) = List.length ks);
  (* [f] is only reached over a non-empty [keys], so the head is there to compare against -- taken
     inside the closure, so an empty [keys] is reported as such rather than raising here. *)
  Verdict.p_all (Printf.sprintf "%s: all device entries carry the same keys in the same order" leg)
    keys ~f:(fun ks -> List.equal String.equal (List.hd_exn keys) ks);
  Verdict.pf "%s: every device entry carries device_name and device_ordinal" leg
    (some_device
    && List.for_all devices ~f:(fun fields ->
        List.exists fields ~f:(fun (k, _) -> String.equal k "device_name")
        && List.exists fields ~f:(fun (k, _) -> String.equal k "device_ordinal")));
  Verdict.pf "%s: device ordinals are 0..n-1 in entry order" leg
    (some_device
    && List.for_alli devices ~f:(fun i fields ->
        match List.Assoc.find fields ~equal:String.equal "device_ordinal" with
        | Some (Sexp.Atom ordinal) -> String.equal ordinal (Int.to_string i)
        | _ -> false))

let () =
  check_negative_controls ();
  let selected = Context.auto () in
  check_dump ~leg:"selected backend" ~backend_name:(Context.backend_name selected)
    (Context.static_properties selected);
  (* [threads] > 1 selects multidev_cc; it needs no hardware, so this leg runs everywhere. *)
  let multidev = Context.cpu ~threads:2 () in
  check_dump ~leg:"multidev_cc" ~backend_name:(Context.backend_name multidev)
    (Context.static_properties multidev)

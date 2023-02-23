open Base
open Ocannl
type nonrec int = int[@@deriving sexp]
let foo (x : int) =
  (Caml.Format.pp_open_hvbox Debug_runtime.ppf 2;
   ((let fname = "test_debug_ppx.ml" in
     Caml.Format.fprintf Debug_runtime.ppf
       "@[\"%s\":%d:%d-%d:%d@ at time UTC@ %s: %s@]" fname 5 18 5 61
       (Core.Time_ns.to_string_utc @@ (Core.Time_ns.now ())) "foo");
    (let fname = "test_debug_ppx.ml" in
     Caml.Format.fprintf Debug_runtime.ppf
       "@[\"%s\":%d:%d-%d:%d@ at time UTC@ %s: %s@]" fname 5 22 5 25
       (Core.Time_ns.to_string_utc @@ (Core.Time_ns.now ())) "");
    Caml.Format.fprintf Debug_runtime.ppf "%s =@ %a@ " "x" Sexp.pp_hum
      (sexp_of_int x));
   (let foo__res = let y : int = x + 1 in 2 * y in
    ((let fname = "test_debug_ppx.ml" in
      Caml.Format.fprintf Debug_runtime.ppf
        "@[\"%s\":%d:%d-%d:%d@ at time UTC@ %s: %s@]" fname 5 28 5 31
        (Core.Time_ns.to_string_utc @@ (Core.Time_ns.now ())) "");
     Caml.Format.fprintf Debug_runtime.ppf "%s =@ %a@ " "foo" Sexp.pp_hum
       (sexp_of_int foo__res));
    Caml.Format.pp_close_box Debug_runtime.ppf ();
    foo__res) : int)
let () = ignore foo

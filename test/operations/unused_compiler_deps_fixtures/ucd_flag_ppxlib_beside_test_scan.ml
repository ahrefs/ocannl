(* Negative control: test_scan is used, the direct ppxlib beside it is not, although test_scan's
   interface loads ppxlib's. *)
let expand = Test_scan.Scan_argv.expand

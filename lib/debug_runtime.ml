open Base

(* FIXME: un-hardcode the path *)
let debug_ch =
  let debug_ch =
    Stdio.Out_channel.create ~binary:false ~append:true "/home/lukstafi/ocannl/debugger.log" in
  Stdio.Out_channel.fprintf debug_ch "\nBEGIN DEBUG SESSION at time UTC %s\n%!"
    (Core.Time_ns.to_string_utc @@ Core.Time_ns.now());
  debug_ch

let ppf =
  let ppf = Caml.Format.formatter_of_out_channel debug_ch in
  Caml.Format.pp_set_geometry ppf ~max_indent:50 ~margin:100;
  ppf

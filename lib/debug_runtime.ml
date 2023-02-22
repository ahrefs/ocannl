open Base

(* FIXME: un-hardcode the path *)
let debug_ch =
  let result =
    Stdio.Out_channel.create ~binary:false ~append:true "/home/lukstafi/ocannl/debugger.log" in
  Stdio.Out_channel.fprintf result "\nBEGIN DEBUG SESSION at time UTC %s\n%!"
    (Core.Time_ns.to_string_utc @@ Core.Time_ns.now());
  result

let ppf = Caml.Format.formatter_of_out_channel debug_ch

(* gh-ocannl-2192: process-isolated production wedge regression. *)
let () = Wait_wedge_fixture.transfer ~cleanup:true ~device_copy:false

module Cu = Cudajit

let () =
  Cu.init ();
  let num_gpus = Cu.device_get_count () in
  Format.printf "\n# GPUs: %d\n%!" num_gpus;
  let gpus = List.init num_gpus (fun ordinal -> Cu.device_get ~ordinal) in
  List.map Cu.device_get_attributes gpus
  |> List.iteri (fun ordinal props ->
         Format.printf "GPU #%d:@ %a@\n%!" ordinal Sexplib0.Sexp.pp_hum @@ Cu.sexp_of_device_attributes props)

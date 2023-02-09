open Base
open Ocannl
let y0 =
  let open Network.O in
    Network.apply
      (Network.apply (+)
         (Network.apply
            (Network.apply ( * )
               (Network.return_term (Operation.number (Float.of_int 2))))
            (Network.return_term (let open Operation.O in !~ "hey"))))
      (Network.return_term (Operation.number (Float.of_int 3)))
let y1 =
  let open Network.O in
    let x_ref = ref [] in
    let x = Network.return (Network.Placeholder x_ref) in
    let body =
      Network.apply
        (Network.apply (+)
           (Network.apply
              (Network.apply ( * )
                 (Network.return_term (Operation.number (Float.of_int 2))))
              (Network.return_term (let open Operation.O in !~ "hey")))) x in
    fun x ->
      x_ref := (x :: (!x_ref));
      (let result__ = Network.unpack body in
       x_ref := (List.tl_exn (!x_ref)); result__)
let y2 =
  let open Network.O in
    let x1_ref = ref [] in
    let x1 = Network.return (Network.Placeholder x1_ref) in
    let x2_ref = ref [] in
    let x2 = Network.return (Network.Placeholder x2_ref) in
    let body =
      Network.apply
        (Network.apply (+)
           (Network.apply (Network.apply ( * ) x1)
              (Network.return_term (let open Operation.O in !~ "hey")))) x2 in
    fun x1 ->
      fun x2 ->
        x1_ref := (x1 :: (!x1_ref));
        x2_ref := (x2 :: (!x2_ref));
        (let result__ = Network.unpack body in
         x1_ref := (List.tl_exn (!x1_ref));
         x2_ref := (List.tl_exn (!x2_ref));
         result__)
let () = ignore (y0, y1, y2)

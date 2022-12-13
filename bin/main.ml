open Base
module F = Ocannl.Formula
open OcamlCanvas.V1

let _ =

  Backend.init ();

  let c = Canvas.createOnscreen ~title:"Hello world" ~pos:(300, 200) ~size:(300, 200) () in

  Canvas.setFillColor c Color.orange;
  Canvas.fillRect c ~pos:(0.0, 0.0) ~size:(300.0, 200.0);

  Canvas.setStrokeColor c Color.cyan;
  Canvas.setLineWidth c 10.0;
  Canvas.clearPath c;
  Canvas.moveTo c (5.0, 5.0);
  Canvas.lineTo c (295.0, 5.0);
  Canvas.lineTo c (295.0, 195.0);
  Canvas.lineTo c (5.0, 195.0);
  Canvas.closePath c;
  Canvas.stroke c;

  Canvas.setFont c "Liberation Sans" ~size:36.0
    ~slant:Font.Roman ~weight:Font.bold;

  Canvas.setFillColor c (Color.of_rgb 0 64 255);
  Canvas.setLineWidth c 1.0;
  Canvas.save c;
  Canvas.translate c (150.0, 100.0);
  Canvas.rotate c (-. Const.pi_8);
  Canvas.fillText c "Hello world !" (-130.0, 20.0);
  Canvas.restore c;

  Canvas.show c;

  let e1 =
    React.E.map (fun { Event.canvas = _; timestamp = _; data = () } ->
        Backend.stop ()
      ) Event.close
  in

  let e2 =
    React.E.map (
      function { Event.canvas = _; timestamp = _;
                 data = { Event.key = key; char = _; flags = _ }; _ } ->
        if phys_equal key KeyEscape then
          Backend.stop ()
      ) Event.key_down
  in

  let e3 =
    React.E.map (fun { Event.canvas = _; timestamp = _;
                       data = { Event.position = (x, y); button = _ } } ->
        Canvas.setFillColor c Color.red;
        Canvas.clearPath c;
        Canvas.arc c ~center:(Float.of_int x, Float.of_int y)
          ~radius:5.0 ~theta1:0.0 ~theta2:Float.(2.0 * Const.pi) ~ccw:false;
        Canvas.fill c ~nonzero:false
      ) Event.button_down
  in

  let frames = ref Int64.zero in

  let e4 =
    React.E.map (fun { Event.canvas = _; timestamp = _; data = _ } ->
        frames := Int64.(!frames + one)
      ) Event.frame
  in

  Backend.run (fun () ->
      ignore e1; ignore e2; ignore e3; ignore e4;
      Caml.Printf.printf "Displayed %Ld frames. Goodbye !\n" !frames)

(* gh-ocannl-1015, gh-ocannl-1081: the provenance-tag reader's rules, put to synthesized sources --
   each spelling it must read beside the nearest text it must not -- and the shipping inventory run
   on a synthetic tree it must accept, then refuse once a citation goes stale, a number collides or
   the phase table lies.

   The inventory skips this file and its golden: the fixture tags are invented, so every one of them
   is a citation no real source mints. The constructor's name is still spelled in pieces ([nv]) so
   that a fixture reads as a fixture. *)
open Base
open Stdio
module Scan = Test_utils.Provenance_tag_scan
open Verdict.Claims

let nv = "Non" ^ "_virtual"
let carriers = [ "Site" ]

let library_with ~store ~consume =
  String.concat ~sep:"\n"
    [
      "let%diagn2_sexp " ^ store ^ " x =";
      "  let exception " ^ nv ^ " of string in";
      "  let helper ~code = if x then raise @@ " ^ nv ^ " code in";
      "  (* \"5:fixture-in-a-comment\" is not a literal *)";
      "  try";
      "    helper ~code:\"9:fixture-helper\";";
      "    if x then raise @@ " ^ nv ^ " \"7:fixture-store\";";
      "    ignore [ \"INFO: 3 things\"; \"7:\"; \":reason\"; \"7:Upper\"; \"7:trailing-\" ]";
      "  with " ^ nv ^ " i -> record (Site i)";
      "";
      "let " ^ consume ^ " y =";
      "  let exception " ^ nv ^ " of string in";
      "  let rec go = function [] -> () | _ :: tl -> go tl in";
      "  try";
      "    go y;";
      "    record (Site \"8:fixture-recorded\");";
      "    raise (" ^ nv ^ " \"4:fixture-consume\")";
      "  with " ^ nv ^ " i -> Error i";
      "";
      "let caller y = match " ^ consume ^ " y with Ok () -> () | Error i -> record (Site i)";
      "";
      "let elsewhere () = record (Tn.Site \"6:fixture-elsewhere\")";
    ]

let library = library_with ~store:"store_check" ~consume:"consume"

let type_source =
  String.concat ~sep:"\n"
    [
      "type provenance = Cap_fixture | Site of string | Refined of provenance * provenance";
      "let rec provenance_to_string = function";
      "  | Cap_fixture -> \"1:cap-fixture\"";
      "  | Site s -> s";
      "  | Refined (a, b) -> provenance_to_string a ^ \" -> \" ^ provenance_to_string b";
    ]

let resolve reads = Scan.merge (fst (Scan.resolve reads))

let read ?(source = "lib/fixture.ml") ?foreign text =
  Scan.read_source ~carriers ?foreign ~source text

let rendered = Scan.renderings ~renderer:"provenance_to_string" ~source:"lib/tnode.ml" type_source
let mints = Scan.merge (rendered @ resolve [ read library ])

let relayed =
  List.filter mints ~f:(fun (m : Scan.mint) -> match m.family with Relayed _ -> true | _ -> false)

let applied =
  List.filter mints ~f:(fun (m : Scan.mint) -> match m.family with Applied _ -> true | _ -> false)

let tags ms =
  List.map ms ~f:(fun (m : Scan.mint) -> m.tag) |> List.dedup_and_sort ~compare:String.compare

let strings = List.equal String.equal

let minter_of tag =
  List.find_map relayed ~f:(fun (m : Scan.mint) -> Option.some_if (String.equal m.tag tag) m.minter)

let mention ?(test_tags = []) text = Scan.mentions ~mints ~test_tags text
let named text = (mention text).named
let stale text = (mention text).stale
let ints = List.equal Int.equal

let table =
  "let phase_table =\n\
  \  [ (\"7:fixture-store\", Store); (\"9:fixture-helper\", Store); (\"4:fixture-consume\", \
   Consumption) ]\n"

let phases =
  [ ("Store", ("lib/fixture.ml", "store_check")); ("Consumption", ("lib/fixture.ml", "consume")) ]

(* Every minting source cites its own tags, as the real ones do. *)
let own_files ms =
  List.map ms ~f:(fun (m : Scan.mint) -> m.source)
  |> List.dedup_and_sort ~compare:String.compare
  |> List.map ~f:(fun path ->
      ( path,
        {
          Scan.named =
            List.filter_map ms ~f:(fun (m : Scan.mint) ->
                Option.some_if (String.equal m.source path) m.number)
            |> List.dedup_and_sort ~compare:Int.compare;
          stale = [];
        } ))

let violations ?malformed ?(mints = mints) ?(pinned = []) ?(files = []) () =
  Scan.violations ?malformed ~mints ~pinned ~files:(own_files mints @ files) ()

let table_violations ?(mints = mints) ?(table = Scan.phase_table table) () =
  Scan.table_violations ~mints ~exn:nv ~table_source:"t.ml" ~table ~phases

let has ~substring = List.exists ~f:(String.is_substring ~substring)

let mint ?(source = "lib/fixture.ml") ?(family = Scan.Applied "Site") ?(minter = "elsewhere") tag =
  { Scan.number = Option.value_exn (Scan.tag_number tag); tag; family; source; minter }

let () =
  (* The families, read off the type. *)
  p "the type's constructors split into rendered, carrier, composite and unread"
    (match
       Scan.type_shape ~type_name:"provenance" (type_source ^ "\ntype other = A | B of int\n")
     with
    | Some s ->
        strings s.rendered [ "Cap_fixture" ]
        && strings s.carriers [ "Site" ] && strings s.composite [ "Refined" ] && strings s.unread []
    | None -> false);
  p "a constructor of another shape is unread, and refused"
    (let shape =
       Scan.type_shape ~type_name:"provenance"
         "type provenance = Site of string | Weird of int | Cap_fixture"
     in
     has ~substring:"constructor Weird carries neither"
       (Scan.family_violations ~identities:[ "Site" ] ~type_source:"t.ml" ~shape ~mints ()));
  p "a nullary constructor mints the tag its renderer case returns"
    (strings (tags rendered) [ "1:cap-fixture" ]);
  p "the renderer must return a carrier's string unchanged"
    (strings (Scan.identity_renderings ~renderer:"provenance_to_string" type_source) [ "Site" ]
    && has ~substring:"does not return Site's string unchanged"
         (Scan.family_violations ~type_source:"t.ml"
            ~identities:
              (Scan.identity_renderings ~renderer:"provenance_to_string"
                 (String.substr_replace_all type_source ~pattern:"Site s -> s"
                    ~with_:"Site s -> \"site:\" ^ s"))
            ~shape:(Scan.type_shape ~type_name:"provenance" type_source)
            ~mints ()));
  (let composed text =
     Scan.composite_renderings ~renderer:"provenance_to_string"
       (String.substr_replace_all type_source
          ~pattern:"provenance_to_string a ^ \" -> \" ^ provenance_to_string b" ~with_:text)
   in
   p "a composite renders each of its provenances, in order, through the renderer"
     (strings (composed "provenance_to_string a ^ \" -> \" ^ provenance_to_string b") [ "Refined" ]);
   p_all "a composite dropping or reordering a provenance is refused"
     [ "provenance_to_string a"; "provenance_to_string b ^ \" -> \" ^ provenance_to_string a" ]
     ~f:(fun text ->
       has ~substring:"does not render each of its provenances"
         (Scan.family_violations ~identities:[ "Site" ] ~composed:(composed text)
            ~type_source:"t.ml"
            ~shape:(Scan.type_shape ~type_name:"provenance" type_source)
            ~mints ())));
  p_empty "a carrier case transforming its string is not rescued by a nested identity case"
    ~over:[ type_source ]
    (Scan.identity_renderings ~renderer:"provenance_to_string"
       (String.substr_replace_all type_source ~pattern:"Site s -> s"
          ~with_:"Site s -> (match other with Site x -> x | _ -> \"site:\" ^ s)"));
  p_all "a composite whose result does not concatenate every provenance, in order, is unread"
    [
      "let _ = provenance_to_string a in provenance_to_string b";
      "if true then provenance_to_string a ^ \" -> \" ^ provenance_to_string b else \"\"";
    ] ~f:(fun text ->
      List.is_empty
        (Scan.composite_renderings ~renderer:"provenance_to_string"
           (String.substr_replace_all type_source
              ~pattern:"provenance_to_string a ^ \" -> \" ^ provenance_to_string b" ~with_:text)));
  p_empty "a guarded or repeated carrier case is not an identity rendering" ~over:[ type_source ]
    (Scan.identity_renderings ~renderer:"provenance_to_string"
       (String.substr_replace_all type_source ~pattern:"| Site s -> s"
          ~with_:"| Site s when false -> s\n  | Site s -> \"site:\" ^ s"));
  p "a nullary constructor with no rendering is refused"
    (has ~substring:"constructor Cap_fixture has no tag rendering"
       (Scan.family_violations ~identities:[ "Site" ] ~type_source:"t.ml"
          ~shape:(Scan.type_shape ~type_name:"provenance" type_source)
          ~mints:applied ()));
  p "a missing type is refused rather than read as no family"
    (has ~substring:"the families cannot be derived"
       (Scan.family_violations ~type_source:"t.ml" ~shape:None ~mints ()));
  p "a carrier applied to no literal anywhere is the reader gone blind"
    (has ~substring:"no source applies Site to a tag literal"
       (Scan.family_violations ~identities:[ "Site" ] ~type_source:"t.ml"
          ~shape:(Scan.type_shape ~type_name:"provenance" type_source)
          ~mints:(rendered @ relayed) ()));
  (* What a relayed code is. *)
  p "every tag literal in a relaying exception's scope is a code, a helper's argument included"
    (strings (tags relayed)
       [ "4:fixture-consume"; "7:fixture-store"; "8:fixture-recorded"; "9:fixture-helper" ]);
  p "a code belongs to the function declaring its exception"
    (List.equal (Option.equal String.equal)
       (List.map
          [ "7:fixture-store"; "9:fixture-helper"; "4:fixture-consume"; "8:fixture-recorded" ]
          ~f:minter_of)
       [ Some "store_check"; Some "store_check"; Some "consume"; Some "consume" ]);
  p_all "the relaying family is named after its exception and its carrier" relayed
    ~f:(fun (m : Scan.mint) -> String.equal (Scan.family_label m.family) ("Site via " ^ nv));
  p "a carrier applied to a literal mints it, qualified or not, inside a scope or outside"
    (strings (tags applied) [ "6:fixture-elsewhere"; "8:fixture-recorded" ]);
  p_none "a tag in a comment is minted by no family" mints ~f:(fun (m : Scan.mint) ->
      String.equal m.tag "5:fixture-in-a-comment");
  (let unrelayed = String.substr_replace_all library ~pattern:"record (Site i)" ~with_:"()" in
   p_none "an exception whose payload no handler hands to a carrier opens no family"
     (resolve [ read unrelayed ])
     ~f:(fun (m : Scan.mint) -> match m.family with Relayed _ -> true | _ -> false));
  (let nested =
     String.concat ~sep:"\n"
       [
         "let outer x =";
         "  let exception " ^ nv ^ " of string in";
         "  let check () = if x then raise @@ " ^ nv ^ " \"2:fixture-in-helper\" in";
         "  let inner () =";
         "    let exception " ^ nv ^ " of string in";
         "    try raise (" ^ nv ^ " \"3:fixture-inner\") with " ^ nv ^ " j -> record (Site j)";
         "  in";
         "  try check (); inner (); raise (" ^ nv ^ " \"7:fixture-store\")";
         "  with " ^ nv ^ " i -> record (Site i)";
       ]
   in
   p "a helper inside a scope mints for the scope, and a scope declared anew inside it is its own"
     (strings
        (List.map
           (resolve [ read ~source:"n.ml" nested ])
           ~f:(fun (m : Scan.mint) -> m.tag ^ " " ^ m.minter))
        [ "2:fixture-in-helper outer"; "3:fixture-inner inner"; "7:fixture-store outer" ]));
  (* A relay belongs to its own scope: a scope is a family only through what ITS handlers do. *)
  (let lone =
     String.concat ~sep:"\n"
       [
         "let first () =";
         "  let exception " ^ nv ^ " of string in";
         "  try raise (" ^ nv ^ " \"2:fixture-first\") with " ^ nv ^ " i -> record (Site i)";
         "let second () =";
         "  let exception " ^ nv ^ " of string in";
         "  try raise (" ^ nv ^ " \"3:fixture-second\") with " ^ nv ^ " i -> log i";
       ]
   in
   p "a same-named scope whose own handler does not relay mints nothing"
     (strings (tags (resolve [ read lone ])) [ "2:fixture-first" ]));
  (let scope handler =
     "let f () =\n  let exception " ^ nv ^ " of string in\n  try raise (" ^ nv
     ^ " \"2:fixture-first\") with " ^ handler
   in
   p_all "a handler whose carrier receives some other value than the payload relays nothing"
     [
       nv ^ " i -> let i = \"x\" in record (Site i)";
       nv ^ " i -> List.iter l ~f:(fun i -> record (Site i))";
       nv ^ " i -> (match o with Some i -> record (Site i) | None -> ())";
       "Other." ^ nv ^ " i -> record (Site i)";
       nv ^ " i -> let* i = next in record (Site i)";
     ]
     ~f:(fun handler -> List.is_empty (resolve [ read (scope handler) ]));
   p_exists "the same handler shape relays when the carrier does receive the payload"
     (resolve [ read (scope (nv ^ " i -> let j = i in ignore j; record (Site i)")) ])
     ~f:(fun (m : Scan.mint) -> String.equal m.tag "2:fixture-first"));
  p "a payload wrapped in a result constructor is relayed by a caller matching it into a carrier"
    (Option.equal String.equal (minter_of "4:fixture-consume") (Some "consume"));
  (let unconsumed =
     String.substr_replace_all library ~pattern:"Error i -> record (Site i)"
       ~with_:"Error i -> log i"
   in
   p_none "a result constructor no caller hands to a carrier relays nothing"
     (resolve [ read unconsumed ])
     ~f:(fun (m : Scan.mint) -> String.equal m.tag "4:fixture-consume"));
  (let unconsumed =
     String.substr_replace_all library ~pattern:"Error i -> record (Site i)"
       ~with_:"Error i -> log i"
   in
   let same_name = "let c y = match consume y with Ok () -> () | Error i -> record (Site i)" in
   let qualified = "let c y = match F.consume y with Ok () -> () | Error i -> record (Site i)" in
   let with_caller text = resolve [ read unconsumed; read ~source:"lib/other.ml" text ] in
   let is_consume (m : Scan.mint) = String.equal m.tag "4:fixture-consume" in
   p_none "a caller of a same-named function of another module relays nothing"
     (with_caller same_name) ~f:is_consume;
   p_exists "a caller qualifying the declaring module through a binding relays"
     (with_caller ("module F = Fixture\n" ^ qualified))
     ~f:is_consume;
   p_none "an unqualified caller of a name the source also binds locally relays nothing"
     (resolve
        [
          read
            (unconsumed
           ^ "\n\
              let other y = let consume = fun _ -> Ok () in\n\
              match consume y with Ok () -> () | Error i -> record (Site i)");
        ])
     ~f:is_consume;
   p_none "a payload wrapped in a result that the handler discards relays nothing"
     (resolve
        [
          read
            (String.substr_replace_all library
               ~pattern:("with " ^ nv ^ " i -> Error i")
               ~with_:("with " ^ nv ^ " i -> ignore (Error i); Error \"unrelated\""));
        ])
     ~f:is_consume;
   p_exists "a caller qualifying the declaring module directly relays"
     (with_caller
        (String.substr_replace_all qualified ~pattern:"F.consume" ~with_:"Fixture.consume"))
     ~f:is_consume);
  (let bad =
     String.substr_replace_all library ~pattern:"\"7:fixture-store\"" ~with_:"\"not-a-tag\""
   in
   p "the relaying exception applied to a string that is no tag is refused, not skipped"
     (List.mem (snd (Scan.resolve [ read bad ])) ("lib/fixture.ml", "not-a-tag") ~equal:Poly.equal));
  (let silent =
     "let f () =\n  let exception " ^ nv ^ " of string in\n  try raise (" ^ nv
     ^ " \"an error message\") with " ^ nv ^ " m -> log m"
   in
   p_empty "a string raised through an exception that relays nothing is no provenance"
     ~over:[ silent ]
     (snd (Scan.resolve [ read silent ])));
  p "a function declaring no exception adds no code to a source that does"
    (strings
       (tags
          (List.filter
             (resolve [ read ("let before () = record (Site \"3:fixture-free\")\n" ^ library) ])
             ~f:(fun (m : Scan.mint) -> match m.family with Relayed _ -> true | _ -> false)))
       (tags relayed));
  (* What citing a tag is. *)
  p "the exception with the number cites the code" (ints (named (nv ^ " 7 fires")) [ 7 ]);
  p "a citation wrapped at the line break still cites it"
    (ints (named ("see [" ^ nv ^ "\n     9] here")) [ 9 ]);
  p "the tag itself cites it, in prose or in a string"
    (ints (named "rejected as 4:fixture-consume, or \"6:fixture-elsewhere\".") [ 4; 6 ]);
  p "the word provenance with a number, emphasized or not, cites any family's tag"
    (ints (named "provenance **6** then provenance 1 then provenance `4`") [ 1; 4; 6 ]);
  p "a tag continued by another tag character, or by a longer number, is a different tag"
    (ints (named "7:fixture-store-twice 17:fixture-store 9:fixture-helper") [ 9 ]
    && strings
         (stale "7:fixture-store-twice 17:fixture-store")
         [ "`17:fixture-store`"; "`7:fixture-store-twice`" ]);
  p "an identifier ending in a spelling word is not that word"
    (ints
       (named ("known_" ^ nv ^ " 7 and X" ^ nv ^ " 7, leading_provenance 7, but " ^ nv ^ " 9"))
       [ 9 ]);
  p "the exception with no number, or with a variable, cites nothing"
    (ints (named (nv ^ " i, " ^ nv ^ " of string, " ^ nv ^ ". Then " ^ nv ^ " 4")) [ 4 ]);
  p "a longer number is not a prefix match on a shorter code"
    (let text = nv ^ " 77 and " ^ nv ^ " 7" in
     ints (named text) [ 7 ] && strings (stale text) [ "`" ^ nv ^ " 77`" ]);
  p "an exception spelling cites only its own family's numbers"
    (strings (stale (nv ^ " 6")) [ "`" ^ nv ^ " 6`" ]);
  p "a family whose last code is retired still has its citations read, and refused"
    (strings
       (Scan.mentions ~spellings:[ nv ] ~mints:(rendered @ applied) ~test_tags:[] (nv ^ " 7")).stale
       [ "`" ^ nv ^ " 7`" ]);
  p "a word-number of a number no tag has is stale"
    (strings (stale "see provenance 3") [ "`provenance 3`" ]);
  p "a bare numeral is not read" (ints (named "refused as 7 first, then as 9:fixture-helper") [ 9 ]);
  p "a tag-shaped token no source mints is a stale citation: a retired tag in prose"
    (strings (stale "the store was refused as 3:fixture-retired.") [ "`3:fixture-retired`" ]);
  p_empty "a test's own tag is a known citation, not a stale one" ~over:[ "99:test-fixture" ]
    (mention ~test_tags:[ "99:test-fixture" ] "set up as 99:test-fixture").stale;
  p_all "colon-joined text that is not a tag is not read"
    [
      "amdgpu 0000:c5:00.0";
      "shape 1:q=1->0:p=1";
      "$1:channels";
      "0:dxg)";
      "a.ml:12:fixture-store";
      "gh-ocannl-12:fixture-store";
      "1.2:fixture-store";
      "7:fixture-storeX";
      "7:one";
    ] ~f:(fun text -> List.is_empty (Scan.tag_tokens text));
  (* The phase table. *)
  p "the table reads as its literal pairs"
    (Option.equal
       (List.equal (fun (a, b) (c, d) -> String.equal a c && String.equal b d))
       (Scan.phase_table table)
       (Some
          [
            ("7:fixture-store", "Store");
            ("9:fixture-helper", "Store");
            ("4:fixture-consume", "Consumption");
          ]));
  p "a table with an element of another shape is unread, not guessed at"
    (Option.is_none (Scan.phase_table "let phase_table = [ (\"7:fixture-store\", phase) ]"));
  p "a source with no table binding has no table"
    (Option.is_none (Scan.phase_table "let other_table = []"));
  p_empty "a table agreeing with the minting functions is accepted" ~over:mints
    (table_violations ());
  p "a table entry placed in the other function's phase is refused"
    (has ~substring:"but it is minted in lib/fixture.ml consume"
       (table_violations
          ~table:(Scan.phase_table "let phase_table = [ (\"4:fixture-consume\", Store) ]")
          ()));
  p "a table entry no raise site mints is refused, a tag of another family included"
    (has ~substring:"is minted nowhere"
       (table_violations
          ~table:(Scan.phase_table "let phase_table = [ (\"6:fixture-elsewhere\", Store) ]")
          ()));
  p "a table entry naming a phase with no minter is refused"
    (has ~substring:"which has no minter"
       (table_violations
          ~table:(Scan.phase_table "let phase_table = [ (\"7:fixture-store\", Cap) ]")
          ()));
  p "an unreadable table is refused rather than skipped"
    (has ~substring:"no `phase_table` binding" (table_violations ~table:None ()));
  p "a phase whose function mints nothing is stale"
    (has ~substring:"names consume in lib/fixture.ml, which mints no"
       (table_violations
          ~mints:
            (List.filter mints ~f:(fun (m : Scan.mint) -> not (String.equal m.minter "consume")))
          ~table:(Scan.phase_table "let phase_table = []")
          ()));
  (* The refusals, and the clean case they must leave alone. *)
  p_empty "a fixture library citing its own tags is accepted" ~over:mints (violations ());
  p "a carrier's argument under a type constraint is read like a bare literal"
    (let r =
       read
         "let t () = record (Site (\"12:fixture-typed\" : string)); record (Site (\"nope\" : \
          string))"
     in
     strings (tags r.mints) [ "12:fixture-typed" ] && strings r.malformed [ "nope" ]);
  p "a carrier applied to a string that is no tag is recorded, and refused"
    (let r = read "let bad () = record (Site \"not-a-tag\"); record (Tn.Site \"12:Bad-tag\")" in
     strings r.malformed [ "not-a-tag"; "12:Bad-tag" ]
     && has ~substring:"which is not a tag"
          (violations ~malformed:(List.map r.malformed ~f:(fun l -> (r.path, l))) ()));
  (let own = "type exemption = Site of string | File\nlet e = Site \"12:fixture-key\"" in
   let aliased =
     "module Scan = Test_utils.Key_scan\n\
      let e = Scan.Site \"12:fixture-key\"\n\
      let t = Tn.Site \"13:fixture-owned\""
   in
   p "a module declaring its own string constructor of a carrier's name is foreign"
     (Scan.declares_own_carrier ~carriers own && not (Scan.declares_own_carrier ~carriers library));
   p_empty "a foreign constructor reached through a chain of module bindings mints nothing"
     ~over:[ aliased ]
     (read ~source:"test/c.ml" ~foreign:[ "Key_scan" ]
        "module A = Test_utils.Key_scan\n\
         module B = A\n\
         module C = C\n\
         let e = B.Site \"12:fixture-key\"")
       .mints;
   p_all "a module name resolves to the binding in scope where it is used"
     [
       "module P = Tnode\n\
        let a = P.Site \"13:fixture-owned\"\n\
        module P = Key_scan\n\
        let b = P.Site \"12:fixture-key\"";
       "module P = Key_scan\n\
        let b = P.Site \"12:fixture-key\"\n\
        module P = Tnode\n\
        let a = P.Site \"13:fixture-owned\"";
       "let b = let module P = Key_scan in P.Site \"12:fixture-key\"\n\
        let a = P.Site \"13:fixture-owned\"";
       "module M = struct module P = Key_scan let b = P.Site \"12:fixture-key\" end\n\
        let a = P.Site \"13:fixture-owned\"";
     ] ~f:(fun text ->
       strings
         (tags (read ~source:"test/d.ml" ~foreign:[ "Key_scan" ] text).mints)
         [ "13:fixture-owned" ]);
   p_all "a local structure declaring its own carrier-named constructor is foreign, inside and out"
     [ "module Local = struct"; "module Local : S = struct" ] ~f:(fun header ->
       strings
         (tags
            (read ~source:"test/e.ml"
               (header
              ^ " type t = Site of string let x = Site \"12:fixture-key\" end\n\
                 let y = Local.Site \"12:fixture-key\"\n\
                 let z = Site \"13:fixture-owned\""))
              .mints)
         [ "13:fixture-owned" ]);
   p_all "an open or include of a foreign module makes the unqualified name foreign in its scope"
     [
       "module Local = struct include Test_utils.Key_scan let b = Site \"12:fixture-key\" end\n\
        let a = Site \"13:fixture-owned\"";
       "let a = Site \"13:fixture-owned\"\nopen Key_scan\nlet b = Site \"12:fixture-key\"";
       "let b = Key_scan.(Site \"12:fixture-key\")\nlet a = Site \"13:fixture-owned\"";
       "let b = let open Key_scan in Site \"12:fixture-key\"\nlet a = Site \"13:fixture-owned\"";
     ] ~f:(fun text ->
       strings
         (tags (read ~source:"test/f.ml" ~foreign:[ "Key_scan" ] text).mints)
         [ "13:fixture-owned" ]);
   p_empty "a foreign constructor applied unqualified in its own module mints nothing" ~over:[ own ]
     (read ~source:"test/support/key_scan.ml" ~foreign:[ "Key_scan" ] own).mints;
   p "a foreign constructor qualified through an alias mints nothing; the owner's still does"
     (strings
        (tags (read ~source:"test/b.ml" ~foreign:[ "Key_scan" ] aliased).mints)
        [ "13:fixture-owned" ]));
  p "a constructor's rendering minted by any other family is refused"
    (has ~substring:"1:cap-fixture is Cap_fixture's rendering, and is also minted by Site"
       (violations ~mints:(Scan.merge (mint "1:cap-fixture" :: mints)) ())
    && has ~substring:"is also minted by Other_cap"
         (violations
            ~mints:(Scan.merge (mint ~family:(Rendered "Other_cap") "1:cap-fixture" :: mints))
            ()));
  (let colliding = Scan.merge (mint "7:fixture-collides" :: mints) in
   p "a new number minted under two tags is refused"
     (has ~substring:"number 7 is minted as 7:fixture-collides and 7:fixture-store"
        (violations ~mints:colliding ()));
   let pin = (7, [ "7:fixture-collides"; "7:fixture-store" ]) in
   p_empty "a pinned collision is accepted" ~over:[ pin ]
     (violations ~mints:colliding ~pinned:[ pin ] ());
   p "a third tag on a pinned number is refused"
     (has
        ~substring:"number 7 is minted as 7:fixture-collides and 7:fixture-more and 7:fixture-store"
        (violations ~mints:(Scan.merge (mint "7:fixture-more" :: colliding)) ~pinned:[ pin ] ()));
   p "a pin that no longer collides is stale"
     (has ~substring:"the pinned collision of 7" (violations ~pinned:[ pin ] ())));
  (let reused = mint ~source:"test/a.ml" "7:test-setup-fixture" in
   p_empty "a test's tag may reuse a library number" ~over:[ reused ]
     (violations ~mints:(Scan.merge (reused :: mints)) ()));
  p "a test constructing a tag no library source mints, not spelled as a test tag, is refused"
    (has ~substring:"test/a.ml constructs 3:fixture-retired, which no library source mints"
       (violations ~mints:(Scan.merge (mint ~source:"test/a.ml" "3:fixture-retired" :: mints)) ()));
  (let cited = mint ~source:"test/a.ml" "6:fixture-elsewhere" in
   p_empty "a test citing a library tag through the carrier is no mint of its own" ~over:[ cited ]
     (violations ~mints:(Scan.merge (cited :: mints)) ()));
  p "a library tag spelled as a test tag is refused"
    (has ~substring:"prefix is reserved for a test's own tags"
       (violations ~mints:(Scan.merge (mint "12:test-fixture" :: mints)) ()));
  p "a minted tag with a one-word reason is refused"
    (has ~substring:"a one-word reason cannot be told"
       (violations ~mints:(Scan.merge (mint "12:fixture" :: mints)) ()));
  p "a code minted in two functions is refused: its provenance cannot name the phase"
    (has
       ~substring:
         "7:fixture-store is minted in lib/fixture.ml consume and lib/fixture.ml store_check"
       (violations
          ~mints:
            (Scan.merge
               (mint
                  ~family:(Relayed { exn = nv; via = "Site" })
                  ~minter:"consume" "7:fixture-store"
               :: mints))
          ()));
  p "a stale citation is refused with its file"
    (has ~substring:"doc.md: `3:fixture-retired` cites a tag no source mints"
       (violations ~files:[ ("doc.md", { Scan.named = []; stale = [ "`3:fixture-retired`" ] }) ] ()));
  p "a minting source whose tags the citation reader misses is the reader gone blind"
    (has ~substring:"but the citation reader does not see it there"
       (Scan.violations ~mints ~pinned:[]
          ~files:[ ("lib/fixture.ml", { Scan.named = [ 7 ]; stale = [] }) ]
          ()));
  p "an excluded prefix no file lives under is stale"
    (List.length
       (Scan.stale_records
          ~records:[ ("docs/proposals/", "r"); ("docs/gone/", "r") ]
          [ "docs/proposals/a.md"; "docs/x.md" ])
    = 1);
  (* The shipping inventory on a synthetic tree. *)
  let exe = Stdlib.Sys.argv.(1) in
  let exe =
    if Stdlib.Filename.is_relative exe then Stdlib.Filename.concat (Stdlib.Sys.getcwd ()) exe
    else exe
  in
  let root = Stdlib.Filename.temp_dir "provenance tag control " "" in
  let write path data = Out_channel.write_all (Stdlib.Filename.concat root path) ~data in
  List.iter [ "arrayjit"; "arrayjit/lib"; "test"; "test/operations"; "docs" ] ~f:(fun dir ->
      Unix.mkdir (Stdlib.Filename.concat root dir) 0o700);
  write "arrayjit/lib/tnode.ml" type_source;
  (* The fixture's comment cites a tag nothing mints, which is the refusal under test below: the
     clean tree cites a minted one there instead. *)
  let low_level =
    String.substr_replace_all
      (library_with ~store:"check_and_store_virtual" ~consume:"instantiate_computations")
      ~pattern:"5:fixture-in-a-comment" ~with_:"4:fixture-consume"
  in
  write "arrayjit/lib/low_level.ml" low_level;
  let boundary = "test/operations/virtual_rejection_boundary.ml" in
  write boundary table;
  write "docs/page.md"
    ("A candidate is refused as `" ^ nv
   ^ " 9`, the store as 7:fixture-store, a cap as provenance 1.\n");
  let run () =
    let out = Stdlib.Filename.temp_file "provenance-tag" ".out" in
    let fd = Unix.openfile out [ Unix.O_WRONLY; Unix.O_TRUNC ] 0o600 in
    let pid = Unix.create_process exe [| exe; "--fixture"; root |] Unix.stdin fd fd in
    let _, status = Unix.waitpid [] pid in
    Unix.close fd;
    let text = In_channel.read_all out in
    Unix.unlink out;
    (status, text)
  in
  let check label ~exit ~message (status, text) =
    let ok =
      (match status with Unix.WEXITED n -> n = exit | _ -> false)
      && String.is_substring text ~substring:message
    in
    if not ok then eprintf "%s captured output:\n%s\n" label text;
    p label ok
  in
  check "shipping inventory lists a page citing tags all three ways" ~exit:0
    ~message:"docs/page.md -- 1 7 9" (run ());
  check "shipping inventory derives the families from the type" ~exit:0 ~message:"  Site via "
    (run ());
  check "shipping inventory groups the codes under the function minting them" ~exit:0
    ~message:
      ("Site via " ^ nv
     ^ " -- arrayjit/lib/low_level.ml, instantiate_computations:\n  4:fixture-consume\n")
    (run ());
  write "docs/page.md" ("A candidate is refused as `" ^ nv ^ " 3`.\n");
  check "shipping inventory refuses a citation of a code no raise site mints" ~exit:1
    ~message:("docs/page.md: `" ^ nv ^ " 3` cites a tag no source mints")
    (run ());
  write "docs/page.md" "The store used to be refused as 3:fixture-retired.\n";
  check "shipping inventory refuses a prose citation of a retired tag" ~exit:1
    ~message:"docs/page.md: `3:fixture-retired` cites a tag no source mints" (run ());
  write "docs/page.md" "";
  write "arrayjit/lib/low_level.ml"
    (low_level ^ "\nlet more () = record (Site \"7:fixture-collides\")\n");
  check "shipping inventory refuses a new number collision" ~exit:1
    ~message:"number 7 is minted as 7:fixture-collides and 7:fixture-store" (run ());
  write "arrayjit/lib/low_level.ml"
    (String.concat ~sep:"\n"
       [
         "let check_and_store_virtual x =";
         "  let exception " ^ nv ^ " of string in";
         "  try helper ~code:\"9:fixture-helper\"; if x then raise @@ " ^ nv
         ^ " \"7:fixture-store\"";
         "  with " ^ nv ^ " i -> record (Site i)";
         "let consumer y =";
         "  match Inliner.instantiate_computations y with Ok () -> () | Error i -> record (Site i)";
         "let elsewhere () = record (Tn.Site \"6:fixture-elsewhere\")";
       ]);
  write "arrayjit/lib/inliner.ml"
    (String.concat ~sep:"\n"
       [
         "let instantiate_computations y =";
         "  let exception " ^ nv ^ " of string in";
         "  try if y then raise (" ^ nv ^ " \"4:fixture-consume\"); Ok () with " ^ nv
         ^ " i -> Error i";
       ]);
  (let moved = run () in
   check "shipping inventory reads a relaying source that never spells the carrier" ~exit:1
     ~message:
       ("Site via " ^ nv
      ^ " -- arrayjit/lib/inliner.ml, instantiate_computations:\n  4:fixture-consume\n")
     moved;
   check "shipping inventory refuses a phase function moved to another source" ~exit:1
     ~message:"but it is minted in arrayjit/lib/inliner.ml instantiate_computations" moved);
  Unix.unlink (Stdlib.Filename.concat root "arrayjit/lib/inliner.ml");
  write "arrayjit/lib/low_level.ml" low_level;
  write boundary "let phase_table = [ (\"4:fixture-consume\", Store) ]\n";
  check "shipping inventory refuses a phase table placing a code in the wrong phase" ~exit:1
    ~message:
      "puts 4:fixture-consume at Store, but it is minted in arrayjit/lib/low_level.ml \
       instantiate_computations"
    (run ());
  let rec remove path =
    if Stdlib.Sys.is_directory path then (
      Array.iter (Stdlib.Sys.readdir path) ~f:(fun name ->
          remove (Stdlib.Filename.concat path name));
      Unix.rmdir path)
    else Unix.unlink path
  in
  remove root

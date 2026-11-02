(** The reading behind [test/operations/label_dispatch_scan] (gh-ocannl-1783): the compiler under
    [arrayjit/lib/] reads a tensor node's label, namespace or names spelled from them, and a
    backend's name, only to show them, to copy a label onto a node it mints, or at a site that says
    why the read decides nothing.

    A node's [label] and [namespace] are public inputs to [Tnode.create]: a user's [{ fragment }]
    tensor has the label head a masked MMA fragment had, and a user tensor can carry any namespace.
    So a compiler decision keyed on them can be forged by a tensor name. The defect recurred as
    [Online_softmax.fold_tile_role] (tile roles by namespace and label, gh-ocannl-1722),
    [Schedule.is_fragment] (a ["fragment"] label head, gh-ocannl-1721) and backend-kind tests by
    substrings of the backend name (gh-ocannl-1723), and earlier as gh-ocannl-793. Each fix replaced
    the spelling with a membership the compiler fills, keyed by the node's uid, or with the
    [Backend_intf] capability facts a backend states.

    The walker is {!Debug_name_scan}'s, read fail-closed: a reference is refused unless it sits in
    an accepted position, so a decision form nobody listed is refused rather than missed. This
    module states the rule's references and its extra accepted positions; the walker's own forms
    (the printed, claim-label, needle, transparent and alias positions, the trust in sink names) are
    in that module's header.

    {1 What it reads}

    Every [.ml] under [arrayjit/lib/], parsed by ppxlib, a [[%string]]'s interpolations included.

    - {b References}: an application of [debug_name] or [label_name] (under any module path; a bare
      one a parameter or a pattern binds is the parameter's value, not the function); [label] and
      [namespace] spelled qualified ([Tn.label], [Tnode.namespace]), or bare where no binding of the
      source is in scope (an [open] brought them in) -- a bare local [label] is a common name and is
      not read; a read of a record field named [label] or [namespace] under any type path
      ([tn.Tn.label], [r.label]) -- fields of other records are read too, the loud direction; and
      any identifier named [backend_name] ([Context.backend_name ctx], [Backends.backend_name b], a
      parameter or a pattern variable named so), read unapplied as the value itself.
    - {b Accepted}: the {!Debug_name_scan} positions (printed or raised, a claim label, a
      [~substring] needle, an alias's body), and:

    + {e composed into a message}: an operand of [^] or [sprintf], or a [[%string]] interpolation:
      the text it is composed into is a message, and the spelling no longer stands alone in it.
      [String.concat] stays transparent, since a label list joined whole is the label;
    + {e a log payload}: anywhere inside a [[%log...]] extension (ppx_minidebug), which is evaluated
      only to be shown;
    + {e raised}: the payload of a [User_error] constructor (under any path);
    + {e copied onto a new node}: the [~label] or [~namespace] argument of [Tn.create] /
      [Tnode.create], a list cons there transparent ([~label:("tile" :: like.Tn.label)]);
    + {e forwarded}: a [~backend_name] labelled argument of any call, whose callee binds the
      parameter under the same name and is judged there;
    + {e marked}: the reference carries [[\@message_only "<reason>"]] (a message by a road the
      walker does not follow, such as a local formatter) or [[\@label_read "<reason>"]] (a read that
      decides nothing a forged name could steer: [Tnode]'s own identity, the user's choice of a
      backend by name, an identity key compared for equality), each with a reason of at least two
      words.

    - {b Refused}: every other reference -- in a [match] or an [if], an argument of [String.equal],
      [String.is_prefix], [=] or any other function, bound by a [let], put in a tuple or a record; a
      reference field destructured in a pattern; a [~backend_name] parameter bound to another name.

    {1 What it deliberately does not read}

    - A decision on composed text ([String.equal ("x" ^ Tn.label tn) "xfragment"]): composition is
      read as display, and a dispatch spelled that way is beyond what this scan guards against.
    - Flow past the positions: a string a [let] binds from an accepted position, and a message built
      and then compared, are not followed. The rule is that a label reaches display, a copied label
      or a marked site, not a data-flow proof.
    - A bare [label] or [namespace] a [let] of the source binds, including [Tnode]'s own top-level
      [label]: inside [tnode.ml] the node's fields are read as fields, which this scan reads.
    - Backend names under other spellings: a backend module's [name], [Device.name], a string a
      [let] binds from [Context.backend_name] under another name (the [let] itself is refused), a
      callee that renames a [~backend_name] parameter is refused at the callee.
    - [Tnode.get_debug_name] and the [code_name] field (the renderer's minted name). *)

let rule =
  {
    Debug_name_scan.references = [ "debug_name"; "label_name"; "backend_name" ];
    parameter_references = [ "backend_name" ];
    qualified_references = [ "label"; "namespace" ];
    values = [ "backend_name" ];
    fields = [ "label"; "namespace" ];
    rule_markers =
      [
        {
          attr = "message_only";
          position = Debug_name_scan.message_only;
          says = "which message the label reaches";
        };
        {
          attr = "label_read";
          position = Debug_name_scan.label_read;
          says = "why the read decides nothing a forged tensor or backend name could steer";
        };
      ];
    raised = [ "User_error" ];
    copy_slots = true;
    composed_is_display = true;
    log_regions = true;
    remedy =
      "a label, namespace or backend name is a display spelling a user's tensor or a backend can \
       forge (gh-ocannl-1783), so decide by uid membership or by the Backend_intf capability \
       facts; the accepted forms are in test/support/label_dispatch_scan.ml";
    node_argument = "the argument of a label reference";
    unparsed_mention = "a label reference";
    marker_targets =
      "a label reference: an application of one to a node, a label or namespace field read, a \
       backend name";
  }

let positions = Debug_name_scan.positions_of rule
let scan ~file source = Debug_name_scan.scan ~rule ~file source

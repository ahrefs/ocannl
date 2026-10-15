(* gh-ocannl-1228: a schedule-cache entry's filename is a fixed-length digest of its key, and the
   entry stores the key it answers.

   The filename used to spell the key out, so its length grew with every key component: on Windows
   without long paths (MAX_PATH, 260 characters for the whole path) every store was refused once the
   cache directory's path plus the key passed it, and on every filesystem a key past the 255-byte
   component limit could never be stored. Pinned here, on synthetic entries in a pure-filesystem
   exercise.

   The entry name's length does not depend on the key's length, so a key long enough to have
   exceeded the component limit stores and replays. The key is stored in the entry:
   [Schedule_cache.entry_keys] reads back exactly the keys stored. Keys the old naming collapsed
   onto one file (a sanitized character; a case difference, on a case-insensitive volume) are
   distinct entries. An entry found under another key's name, which is what a digest collision looks
   like, is a miss the lookup decided rather than a refusal, and the entry it belongs to stays
   readable. Last, [Atomic_file]'s staging-open refusal names a path's length past Windows' limit,
   on Windows. *)

open Base
module SC = Ir.Schedule_cache
module AF = Utils.Atomic_file
open Verdict.Claims
module Cache_plant = Test_utils.Cache_plant

let entry backend =
  Cache_plant.entry ~backend ~source_digest:"gh1228-source" ~best_ms:1. ~baseline_ms:2. ()

let cache_dir = "autotune_cache_entry_name"

let clean_dir dir =
  if Stdlib.Sys.file_exists dir && Stdlib.Sys.is_directory dir then (
    Array.iter (Stdlib.Sys.readdir dir) ~f:(fun name ->
        Stdlib.Sys.remove (Stdlib.Filename.concat dir name));
    Stdlib.Sys.rmdir dir)

let name_length key = String.length (Stdlib.Filename.basename (SC.cache_file ~dir:cache_dir ~key))

(* The backend field is the payload's own marker of which key it was stored under. *)
let replays key =
  match SC.lookup ~dir:cache_dir ~key:(Some key) with
  | Some e -> String.equal e.SC.backend key
  | None -> false

let refusals f =
  let result, io = SC.recording_cache_io f in
  (result, List.filter_map io ~f:(fun (r : SC.cache_io) -> r.SC.refusal))

let () =
  clean_dir cache_dir;
  (* A long key is a realistic one grown further: every component a key adds lengthens it. *)
  let long_key = String.concat ~sep:"-" (List.init 120 ~f:(fun i -> Printf.sprintf "c%04d" i)) in
  let keys = [ "k"; "gh1228-short-key"; String.prefix long_key 300; long_key ] in
  let lengths = List.map keys ~f:name_length in
  p "the keys span the 255-byte component limit"
    (List.exists keys ~f:(fun k -> String.length k < 64)
    && List.exists keys ~f:(fun k -> String.length k > 255 * 2));
  p_all "every entry name has the same length, whatever its key's length" lengths ~f:(fun l ->
      l = List.hd_exn lengths);
  p_all "every entry name is a .sexp file in the cache directory" keys ~f:(fun key ->
      let file = SC.cache_file ~dir:cache_dir ~key in
      String.equal (Stdlib.Filename.dirname file) cache_dir && String.is_suffix file ~suffix:".sexp");
  let (), store_refusals =
    refusals (fun () ->
        List.iter keys ~f:(fun key -> SC.store ~dir:cache_dir ~key:(Some key) (entry key)))
  in
  p_empty "storing under every key, the longest included, is refused nowhere" ~over:keys
    store_refusals;
  p_all "every stored key replays its own entry" keys ~f:replays;
  p "the entries store their keys: entry_keys reads back exactly the keys stored"
    (List.equal String.equal
       (List.sort (SC.entry_keys ~dir:cache_dir) ~compare:String.compare)
       (List.sort keys ~compare:String.compare));

  (* The old naming mapped every character outside [A-Za-z0-9_-] to [_], and a case-insensitive
     volume folds case: each pair below was one file. *)
  let twins = [ ("gh1228.dot", "gh1228_dot"); ("gh1228-Case", "gh1228-case") ] in
  List.iter twins ~f:(fun (a, b) ->
      SC.store ~dir:cache_dir ~key:(Some a) (entry a);
      SC.store ~dir:cache_dir ~key:(Some b) (entry b));
  p_all "keys the old naming collapsed onto one file stay distinct entries" twins ~f:(fun (a, b) ->
      replays a && replays b);

  (* A digest collision, simulated: another key's complete entry at this key's name. *)
  let owner = "gh1228-owner" and intruder = "gh1228-collides" in
  SC.store ~dir:cache_dir ~key:(Some owner) (entry owner);
  Stdio.Out_channel.write_all
    (SC.cache_file ~dir:cache_dir ~key:intruder)
    ~data:(Stdio.In_channel.read_all (SC.cache_file ~dir:cache_dir ~key:owner));
  let collided, collision_io =
    SC.recording_cache_io (fun () -> SC.lookup ~dir:cache_dir ~key:(Some intruder))
  in
  p "another key's entry under this key's name is a miss" (Option.is_none collided);
  p "the collision miss is the lookup's own decision, not a refusal"
    (match collision_io with
    | [ { SC.op = SC.Lookup; key; refusal = None; _ } ] -> String.equal key intruder
    | _ -> false);
  p "the entry the colliding file belongs to still replays" (replays owner);
  clean_dir cache_dir;

  (* The staging-open diagnosis (gh-ocannl-1228 part 2): Windows refuses an over-long path as
     ENOENT, so the refusal names its length. Rooted paths, so the working directory adds
     nothing. *)
  let root = if Stdlib.Sys.win32 then "C:\\" else "/" in
  let rooted n = root ^ String.make (n - String.length root) 'a' in
  let over = rooted 300 and at_limit = rooted 259 in
  p "on Windows, a path past 259 characters gets a note naming its length"
    (match AF.windows_path_length_note ~win32:true over with
    | Some note -> String.is_substring note ~substring:"300 characters"
    | None -> false);
  p "on Windows, a path of 259 characters gets no note"
    (Option.is_none (AF.windows_path_length_note ~win32:true at_limit));
  p "elsewhere, no path gets the note"
    (Option.is_none (AF.windows_path_length_note ~win32:false over));
  (* A relative name that resolves past the limit only with the working directory counted: 260
     characters resolved where the directory is shorter than that, and never under one character, so
     a deeper working directory still yields a name rather than a negative length. *)
  let cwd_prefix = String.length (Stdlib.Filename.concat (Stdlib.Sys.getcwd ()) "") in
  let relative_length = Int.max 1 (260 - cwd_prefix) in
  p "a relative path is measured as resolved against the working directory"
    (match AF.windows_path_length_note ~win32:true (String.make relative_length 'r') with
    | Some note ->
        String.is_substring note
          ~substring:(Int.to_string (cwd_prefix + relative_length) ^ " characters")
    | None -> false)

(** Publishing a file so that no reader ever observes a half-written one (gh-ocannl-780).

    Every writer here stages its bytes into a uniquely named sibling of the target and then renames
    that sibling over it. A reader of the target therefore sees the complete previous content or the
    complete new content, never an intention in progress, and never a missing file — the target is
    never unlinked, truncated or opened for writing.

    The dance has three parts a hand-written copy tends to get only two of:

    - {b Uniqueness.} A fixed [<path>.tmp] is shared state: two concurrent writers to the same
      target write into one another's staging file and commit a mixture. The staging name here
      carries the writer's pid, a per-process counter and a nonce, and the file is created
      EXCLUSIVELY — a name already taken is retried rather than opened — so uniqueness holds even
      between hosts or pid namespaces sharing one filesystem, where pids and counters collide.
    - {b Failure cleanup.} A staging file is removed on every path out of {!with_channel} that is
      not a successful commit, so a failed write leaves neither a torn target nor an accumulating
      intention.
    - {b Crash-stale cleanup.} A writer killed between staging and commit cannot clean up after
      itself, and its staging file would otherwise sit in the directory forever. {!cleanup_stale}
      and {!cleanup_stale_for} remove staging files older than an age no live attempt can plausibly
      reach; {!cleanup_stale_once} is the once-per-process-per-directory form the cache writers
      call.

    {2 Windows}

    The portable dance is not the POSIX one, and the differences are measured rather than inferred
    (gh-ocannl-588):

    - Replacing the target by [rename] is right, and truncating it in place is wrong, on Windows
      most of all: a live memory mapping of the target blocks neither the rename nor a delete
      (modern NTFS has POSIX delete semantics), but it does pin the file's SIZE, so reopening the
      path for writing — the obvious "just overwrite it" alternative — is the one operation Windows
      refuses outright ([ERROR_USER_MAPPED_FILE]). Nothing here ever opens the target.
    - The staging file is closed before it is renamed and before it is removed. The C runtime opens
      files without [FILE_SHARE_DELETE], so on Windows both operations fail while this process still
      holds the handle.
    - For the same sharing reason a rename can fail while ANOTHER process, or another domain, has
      the target open for reading. The commit polls for up to a second before propagating the
      failure; on POSIX the first attempt always succeeds and the retry costs nothing. A second is
      not a guarantee and is not meant as one: a reader that reopens the target as fast as it closes
      it leaves no window for the rename at all, and against one of those the publish is REFUSED —
      so a caller on Windows must be prepared for {!write_all} and {!with_channel} to raise
      [Sys_error] where no POSIX caller ever would. What it may rely on is that a refusal changes
      nothing: the target still holds a complete payload, and the staging file is cleaned up.
    - Staging names are matched caselessly, because on Windows and on a default macOS volume two
      spellings of one name address one file. See {!is_staging_file_for}. *)

val staging_infix : string
(** The infix marking a staging file: [<stem>.ocannl-stage.<pid>.<counter>.<nonce>], where the stem
    is derived from the target's basename (see {!is_staging_file}) and the three fields that follow
    are FIXED-WIDTH lowercase hex — eight digits, eight digits, sixteen digits. Distinctive rather
    than generic, so a sweep can attribute leftovers; fixed-width so that a [.gitignore] glob, which
    has no way to spell "one or more digits", can describe exactly this set and no ordinary file
    besides. *)

val field_width : int
(** Width of the pid and counter fields in a staging name. Exposed with {!nonce_width} so the test
    can derive the repository's [.gitignore] rule from the generator's constants rather than
    restating them. *)

val nonce_width : int
(** Width of the nonce field in a staging name. See {!field_width}. *)

val is_staging_file : string -> bool
(** Whether a file NAME (not necessarily a path) is one of this module's staging artifacts. The
    predicate to be checked against, rather than a second spelling of the naming scheme.

    It recognizes the WHOLE generated name — a non-empty stem, the infix, then the three fixed-width
    hex fields — not merely the presence of the infix, and each field at exactly the width and
    alphabet generation uses, over a stem no longer than generation can emit. The sweep deletes what
    this accepts, so a file someone else named [report.ocannl-stage.backup] must not be accepted,
    and neither must one whose fields are hex of some other width or whose stem is longer than the
    budget. Every value this accepts is one the generator can produce: the two are written from the
    same widths, and the nonce is a full 64-bit draw rather than a masked one.

    The stem is the target's basename where that fits, and a truncation of it plus a digest of the
    whole where it does not: a staging name must stay inside the filesystem's per-component limit
    however long the name being published is.

    The repository's [.gitignore] rule is the same shape, and is exact except for the stem's length
    bound, which a glob has no way to state. That looseness hides a file from [git status]; it never
    deletes one, because deletion consults this predicate. *)

val is_staging_file_for : path:string -> string -> bool
(** {!is_staging_file} narrowed to the staging artifacts of ONE published file: whether the NAME is
    a staging file whose stem is the one [path] generates. For a directory OCANNL shares with
    unrelated content — a checkpoint next to the user's other files — where another publication's
    artifact is neither this caller's to judge nor to delete.

    The comparison is caseless. On Windows and on a default macOS volume [Model.bin] and [model.bin]
    are the same file, so a case-sensitive match would leave a model-sized artifact unreclaimed;
    where paths really are case-sensitive, the only effect is that a save also reclaims a
    case-twin's hour-old abandoned staging file, which the directory-wide sweep would remove anyway.

    The folding is ASCII, which is not the filesystem's: NTFS folds by an upcase table fixed when
    the volume was formatted and APFS by its own Unicode version, so [Ä.bin] and [ä.bin] are one
    file there and two names here. The consequence is bounded and is a LEAK, never a wrong deletion
    — the narrow sweep declines to reclaim a non-ASCII case-twin's artifact, while {!cleanup_stale},
    which compares no stems, still reclaims it wherever OCANNL owns the directory. Matching the
    filesystem exactly would mean carrying a Unicode table and still disagreeing with some volume,
    so it is left as a known bound rather than approximated (Codex P2, round 8). *)

val windows_path_length_note : ?win32:bool -> string -> string option
(** [Some note] when [path], resolved against the working directory if relative, is longer than the
    259 characters Win32 accepts without [LongPathsEnabled], on Windows ([?win32] defaults to
    [Sys.win32]); [None] otherwise. Such a path is refused as [ENOENT], "No such file or directory",
    which names nothing about length, so a refused staging open appends this note to its [Sys_error]
    (gh-ocannl-1228). Exposed with [?win32] so the note can be tested on any host. *)

val ensure_dir : string -> unit
(** Creates the directory and its missing parents, tolerating concurrent creators. A no-op for
    ["."], ["/"] and the empty string, so a bare filename's [Filename.dirname] is safe to pass. *)

val publish_staged : staging:string -> path:string -> unit
(** [publish_staged ~staging ~path] atomically renames a complete, closed staging path over [path],
    using the same bounded Windows retry as file publication. It also accepts a staged directory
    tree, which lets callers build a multi-file artifact privately before exposing the whole tree.

    The caller owns staging-name uniqueness, ensures both paths are on the same filesystem, and
    removes [staging] after a failed publication. Prefer {!with_channel} for one file: it provides
    all three automatically. *)

val with_channel :
  ?before_commit:(unit -> unit) ->
  ?binary:bool ->
  path:string ->
  f:(Stdlib.out_channel -> 'a) ->
  unit ->
  'a
(** [with_channel ~path ~f ()] creates a fresh staging file next to [path], calls [f] with a channel
    onto it, closes the channel, renames the staging file over [path], and returns [f]'s result.
    This is the primitive; {!write_all} is the in-memory form. The channel is opened in binary mode
    unless [~binary:false] is passed.

    [path]'s directory must exist — publishing is not directory management, and creating one here
    would turn a write to a mistyped path into a success nobody looks at. Call {!ensure_dir} first
    where the directory is the caller's to create.

    [?before_commit] runs after the channel is closed and before the rename. It is the seam a caller
    uses to observe or to fail the window in which the payload is staged but not yet committed — the
    resource fault-injection points do exactly that.

    Any exception from [f], from [before_commit] or from the rename closes the channel, removes the
    staging file, and is re-raised with its original backtrace. The target is left exactly as it
    was.

    Every FILESYSTEM refusal — a missing or unwritable directory, an exhausted disk, a rename the
    platform declines past its retries — reaches the caller as [Sys_error], whichever operation
    refused it. That the staging file is opened through [Unix] (which is what makes its creation
    exclusive) is an implementation detail and does not change what a caller catches, so a
    best-effort writer like [Schedule_cache.store] needs one handler and not a taxonomy. Exceptions
    raised by [f] and [before_commit] are the caller's own and pass through unchanged. *)

val write_all :
  ?before_commit:(unit -> unit) -> ?binary:bool -> path:string -> data:string -> unit -> unit
(** {!with_channel} for a payload already in memory. Every argument is labeled and the call is
    closed by [()], so [?before_commit] can be passed in whichever position reads best at the call
    site. *)

val default_max_age_seconds : float
(** One hour: orders of magnitude above the milliseconds a staging window lasts, so a file this old
    belongs to a writer that died, not to one still running. *)

val cleanup_stale : ?max_age_seconds:float -> string -> unit
(** [cleanup_stale dir] removes the staging files in [dir] whose modification time is older than
    [?max_age_seconds] (default {!default_max_age_seconds}). Everything is best-effort: an
    unreadable directory, a file that vanished under the sweep, or a file another user owns is
    skipped rather than reported. Files that are not staging artifacts are never touched.

    The threshold is on INACTIVITY, not on age since creation: a writer's own writes advance the
    staging file's mtime, so a publication that takes hours to stream is never mistaken for an
    abandoned one, and the mtime is re-read at the moment of removal so a writer that resumed during
    the sweep is spared. What remains outside that is a writer STALLED with no I/O for the whole
    threshold — indistinguishable from a dead one by any portable signal, since a file lock is
    released at the close that precedes the commit and pid probing is unavailable on Windows and
    defeated by pid reuse. Its worst outcome is bounded by design: staging names are unique, so the
    sweep can never reach another writer's attempt, and the stalled writer's own commit then fails
    with [Sys_error] against an intact target — a publication that does not happen, never a torn or
    lost file.

    The whole-directory scope suits a directory OCANNL owns — a schedule cache, the per-user probe
    cache. For one published file inside a directory it does not own, use {!cleanup_stale_for}. *)

val cleanup_stale_for : ?max_age_seconds:float -> string -> unit
(** [cleanup_stale_for path] is {!cleanup_stale} over [path]'s directory, narrowed to the staging
    artifacts of [path] itself. This is what a writer publishing into a directory of somebody else's
    files calls before publishing: a checkpoint killed between streaming and commit leaves a staging
    file the size of the model, and the next save of that same checkpoint is the event that can
    reclaim it. *)

val cleanup_stale_once : ?max_age_seconds:float -> string -> unit
(** {!cleanup_stale} at most once per directory per process, for callers that would otherwise sweep
    on every write. Thread-safe across domains. A directory that does not exist yet is not recorded
    as swept: a cache's reader reaches here before its first writer creates the directory, and
    spending the process's one sweep on nothing would leave the first crash-stale file until the
    next process. *)

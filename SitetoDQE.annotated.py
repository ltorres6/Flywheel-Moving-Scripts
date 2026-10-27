# SitetoDQE.py, optimized, with comments on what changed from your version and why.
# The code is identical to SitetoDQE.py: only the '# >>>' comment lines are added.
# Ids match report.html: C = performance change, F = fix or behavior change.
#
#   C1  Create without the follow-up read
#   C2  Rename and move in one update
#   C3  Use what you already read
#   C4  Remember destination subjects and create them only when needed
#   C5  Look up the destination projects once
#   C6  Remember destination sessions and their acquisition labels
#   C7  List only the source group's projects
#   C8  Read with small calls and page long lists
#   C9  Process sessions in parallel (WORKERS, default 4)
#   F1  Stops when project stats aren't ready
#   F2  Retry wrapper never retries
#   F3  Rename suffixes can repeat
#   F4  Subject lookup in the Patient Information handler
#   F5  Empty containers
#   F6  One failure stops the whole run
#   F7  Configuration

"""Move site submissions out of the study projects for review.

  Patient Information (read Complete)  -> <source group>/Participant History
  Visit 1 OCT / FP acquisitions        -> <source group>/DQE
  Uploader Acknowledgement             -> <certification group>/Certification Review
  Certification                        -> <certification group>/Certification Review

Source sessions left without acquisitions are deleted.

Sessions are processed by WORKERS threads (default 4; WORKERS=1 runs them one at a
time). Each session is read, planned and moved by one thread, so the order inside
a session is kept: its moves land before its delete. Moves into the same
destination happen one at a time, and a destination label is only recorded once
the move that uses it has succeeded.

Reads use small per-container calls (a session's acquisitions, one session) rather
than project-wide filtered listings. Those listings get slower per call as projects
grow; the small calls stay flat. See report.html.

Reader tasks are created with your utils/tasks.py exactly where the original
script creates them (after each Visit 1 move; for an OCT original/companion pair,
only for the companion).

If a session fails, the error is printed and the run continues. The script exits
non-zero at the end in that case. Re-running it picks up whatever is left.
Ctrl-C cancels the sessions that have not started and lets the running ones finish.
"""

import os
import re
import sys
import threading
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

# >>> [F2] New import. backoff replaces your flywheel_modify_with_retry wrapper (see
# >>>      _retry_on_500 below). It is already installed with fw-client, which your
# >>>      script uses.
import backoff
import flywheel
from flywheel.models import ContainerDeleteReason
from fw_client import FWClient
# >>> [unchanged] Unchanged: reader tasks are created with your own utils/tasks.py, with
# >>>             the same FWClient as before.
from utils.tasks import create_task  # local file, unchanged

# Configuration
# >>> [F7] Same key file as your script. FW_API_KEY_PATH or FW_API_KEY can point
# >>>      elsewhere, and SRC_GROUP / CERT_GROUP can change the groups. The script prints
# >>>      these settings when it starts instead of clearing the terminal.
API_KEY_PATH = os.environ.get(
    "FW_API_KEY_PATH", "/home/myounis/copysubjectscript/api_key.txt"
)
SOURCE_GROUP = os.environ.get("SRC_GROUP", "abiduw")
CERTIFICATION_GROUP = os.environ.get("CERT_GROUP", "certification")
EXCLUDED_PROJECTS = {"Completed", "DQE", "Participant History", "Archive"}
EXCLUDED_INITIALS = {"FP", "FA", "OCT", "FAF", "RE", "LE", "OU", "CERT"}
# >>> [C9] How many sessions are handled at the same time. WORKERS=1 handles one session
# >>>      at a time, like your script, with the lowest load on the server.
WORKERS = int(os.environ.get("WORKERS") or 4)
# >>> [C8] Sessions are read 100 at a time instead of in one listing of the whole
# >>>      project.
PAGE_SIZE = 100

# Reader task assignment. Copy these four values from your current
# handle_visit1_session (the "#modify this" lines). Empty values create tasks
# with no protocol or assignee; the script warns about that when it starts.
# >>> [unchanged] Your four reader-task values, moved here from handle_visit1_session so
# >>>             they are in one place. Copy your current values in; the script warns at
# >>>             start if any is empty.
OCT_ASSIGNEE = ""  # modify this
OCT_PROTOCOL_LABEL = ""  # modify this
FP_ASSIGNEE = ""  # modify this
FP_PROTOCOL_LABEL = ""  # modify this


def api_key():
    """(key, source): FW_API_KEY if set, otherwise the key file, as in the original script."""
    if os.environ.get("FW_API_KEY"):
        return os.environ["FW_API_KEY"], "the FW_API_KEY environment variable"
    with open(API_KEY_PATH) as f:
        return f.read().strip(), API_KEY_PATH


def extract_initials(filename):
    """Tech initials from a certification file name, e.g. CERT_JD_photos.zip -> JD."""
    for token in re.split(r"[_\W]+", filename):
        if re.fullmatch(r"[A-Z]{2,3}", token) and token not in EXCLUDED_INITIALS:
            return token
    return None


def extract_prefix(project_label):
    """'101 - Alpha Study' -> '101 - Alpha'."""
    number, rest = project_label.split(" - ", 1)
    return f"{number} - {rest.split()[0]}"


# >>> [F3] Replaces the millisecond suffix (int(time.time() * 1000) % 1000), which can
# >>>      repeat. The first free _2, _3, ... is checked against the labels already in
# >>>      the destination.
def free_label(label, taken):
    """label, or label_2, label_3, ... whichever is not in taken."""
    if label not in taken:
        return label
    n = 2
    while f"{label}_{n}" in taken:
        n += 1
    return f"{label}_{n}"


# The SDK already retries 429/502/503/504 and connection errors, but not 500.
# Flywheel can return a one-off 500 on a read that succeeds on the next try.
# Only reads go through this; they are safe to repeat.
# >>> [F2] Your retry wrapper caught requests errors, which the current Flywheel SDK no
# >>>      longer raises, so it never retried. The SDK retries busy and unavailable
# >>>      responses on its own. This adds a retry for an occasional server error (500),
# >>>      for reads only: a create that timed out may already exist, so creates are not
# >>>      repeated.
_retry_on_500 = backoff.on_exception(
    backoff.expo,
    flywheel.ApiException,
    max_tries=4,
    giveup=lambda e: getattr(e, "status", None) != 500,
)


def read(fn, *args, **kwargs):
    return _retry_on_500(fn)(*args, **kwargs)


# >>> [C4] [C6] New. Your script searched for the destination subject and session, and
# >>>           listed the destination's acquisitions, before every move (move_acq,
# >>>           rename_and_move_acquisition and the handlers). This class looks each one
# >>>           up once per run and keeps it, so a move needs one request instead of four
# >>>           or five.
class Destinations:
    """Destination containers, looked up once per run and remembered.

    Subjects, a subject's sessions and a session's acquisition labels are each
    fetched at most once. Containers created here are recorded as empty so they
    are never fetched. Moves into a destination run under that destination's
    lock, and the label they use is recorded only after the move succeeds, so
    two threads never pick the same label and a failed move leaves no trace.
    """

    def __init__(self, fw):
        self.fw = fw
        self._subjects = {}  # (project_id, label) -> subject
        self._sessions = {}  # subject_id -> {label: session}
        self._acq_labels = {}  # session_id -> set of acquisition labels
        self._full = {}  # session_id -> full session model (for create_task)
        self._guard = threading.Lock()
        self._locks = defaultdict(threading.Lock)

    def _lock(self, key):
        with self._guard:
            return self._locks[key]

    def subject(self, project, label):
        key = (project.id, label)
        with self._lock(key):
            if key not in self._subjects:
                # >>> [C4] [F4] One lookup path for destination subjects. It replaces
                # >>>           find_one in move_acq, which only worked because main()
                # >>>           created every subject in DQE first, and the find_first /
                # >>>           find_one / add_subject sequence in
                # >>>           handle_patient_information_session.
                subject = read(project.subjects.find_first, f'label="{label}"')
                if subject is None:
                    print(f"Creating subject '{label}' in {project.label}")
                    # fw.add_subject returns only the id; project.add_subject would also re-fetch it
                    # >>> [C1] [F5] fw.add_subject() returns the new id in one request;
                    # >>>           project.add_subject() also reads the subject back.
                    # >>>           Subjects are created only when something moves to
                    # >>>           them, so DQE no longer gets an empty subject for
                    # >>>           every source subject.
                    sid = self.fw.add_subject({"project": project.id, "label": label})
                    subject = SimpleNamespace(id=sid, label=label, project=project.id)
                    self._sessions[sid] = {}
                self._subjects[key] = subject
            return self._subjects[key]

    def _sessions_of(self, subject):
        """{label: session} for a destination subject. Callers hold the subject's lock."""
        if subject.id not in self._sessions:
            sessions = {}
            for s in read(self.fw.get_subject_sessions, subject.id):
                sessions.setdefault(s.label, s)  # first match wins, as in the original
            self._sessions[subject.id] = sessions
        return self._sessions[subject.id]

    def session(self, subject, label):
        """Existing destination session with this label, created if missing."""
        with self._lock(subject.id):
            sessions = self._sessions_of(subject)
            if label not in sessions:
                print(f"Creating session '{label}' under subject '{subject.label}'")
                # >>> [C1] Same for sessions: one request, and a new session is known to
                # >>>      be empty, so its acquisitions are never listed.
                sid = self.fw.add_session(
                    {"project": subject.project, "subject": subject.id, "label": label}
                )
                sessions[label] = SimpleNamespace(id=sid, label=label)
                self._acq_labels[sid] = set()
            return sessions[label]

    def move_session(self, session, subject):
        """Move a whole session under subject, renaming it if the label is taken there."""
        with self._lock(subject.id):
            self._move_session_locked(session, subject)

    # >>> [C9] Uploader Acknowledgement: the same rule as
    # >>>      handle_uploader_acknowledgement_session. The first session for a
    # >>>      Certification Review subject moves whole and later ones merge into it.
    # >>>      Checking and moving happen together, so two sessions arriving at the same
    # >>>      time still merge.
    def merge_target_or_move(self, session, subject):
        """The destination session to merge into, or None after moving the session whole.

        The check and the move happen under the subject's lock, so two sessions
        arriving together still end up merged, as when the original handles them
        one after the other.
        """
        with self._lock(subject.id):
            existing = self._sessions_of(subject).get(session.label)
            if existing is not None:
                return existing
            self._move_session_locked(session, subject)
            return None

    # >>> [C2] [F3] Session moves: your script renamed the session, read it back, then
    # >>>           moved it (three requests). Here the new label, if one is needed, goes
    # >>>           in the same update as the move.
    def _move_session_locked(self, session, subject):
        sessions = self._sessions_of(subject)
        label = free_label(session.label, sessions)
        body = {"subject": subject.id}
        if label != session.label:
            print(
                f"Session '{session.label}' already exists in destination, renaming to '{label}'"
            )
            body["label"] = label
        print(f"Moving session '{session.label}' to subject '{subject.label}'")
        self.fw.modify_session(session.id, body)
        sessions[label] = session  # recorded only once the move succeeded

    # >>> [C2] [C6] Replaces rename_and_move_acquisition. Your version listed the
    # >>>           destination's acquisitions, renamed the acquisition, read it back and
    # >>>           then moved it. Here the destination's labels are kept from the first
    # >>>           listing, and the label and the move go in one update. A label is
    # >>>           recorded only after the move succeeds.
    def move_acquisition(self, acq, dest_session, label):
        """Move acq into dest_session as label (label_2, ... if taken): rename and move in one update."""
        with self._lock(dest_session.id):
            if dest_session.id not in self._acq_labels:
                self._acq_labels[dest_session.id] = {
                    a.label
                    for a in read(self.fw.get_session_acquisitions, dest_session.id)
                }
            labels = self._acq_labels[dest_session.id]
            new_label = free_label(label, labels)
            if new_label != acq.label:
                print(f"Acquisition '{acq.label}' will be renamed to '{new_label}'")
            print(f"Moving acquisition '{new_label}' to session '{dest_session.label}'")
            self.fw.modify_acquisition(
                acq.id, {"session": dest_session.id, "label": new_label}
            )
            labels.add(new_label)  # recorded only once the move succeeded
            acq.label = new_label  # create_task gets the renamed acquisition, as in the original

    # >>> [unchanged] create_task receives the full destination session, as in your
    # >>>             script (which passed a reloaded session). It is read once per
    # >>>             destination session instead of before every move.
    def full_session(self, session):
        """The full session model, fetched once, for create_task (the original passed a reloaded session)."""
        with self._lock(("full", session.id)):
            if session.id not in self._full:
                self._full[session.id] = read(self.fw.get_session, session.id)
            return self._full[session.id]


class Planner:
    """Decides, per source session, what moves where. Destinations does the moving."""

    def __init__(self, fw, destinations):
        self.fw = fw
        self.dest = destinations
        self._projects = {}
        self._projects_lock = threading.Lock()

    # >>> [C5] DQE, Participant History and Certification Review are looked up the first
    # >>>      time they are needed and kept. Your script looked up DQE once per study
    # >>>      project and Participant History once per Patient Information session.
    def project(self, path):
        """Destination project, looked up on first use (as in the original) and remembered."""
        with self._projects_lock:
            if path not in self._projects:
                self._projects[path] = read(self.fw.lookup, path)
            return self._projects[path]

    # >>> [unchanged] Same rules as handle_patient_information_session: skipped without a
    # >>>             viewer read, moved when the first reader's status is Complete,
    # >>>             renamed if the label already exists in Participant History.
    def patient_information(self, session, subject_label, acqs):
        viewer = (session.info or {}).get("ohifViewer")
        if not isinstance(viewer, dict) or "read" not in viewer:
            return [], False  # no viewer read yet: skip quietly, as the original does
        reads = viewer["read"] or {}
        first = next(iter(reads.values()), None)
        if not first or str(first.get("readStatus")) != "Complete":
            print(
                f"Skipping {subject_label} / {session.label}: patient info is not complete"
            )
            return [], False
        history = self.project(f"{SOURCE_GROUP}/Participant History")
        dest_subject = self.dest.subject(history, subject_label)
        return [("session", session, dest_subject)], True

    # >>> [C3] Same selection as handle_visit1_session: an OCT original moves with the
    # >>>      first remaining non-original OCT, lone OCT and FP acquisitions move on
    # >>>      their own, and tasks are created for the same acquisitions. Your script
    # >>>      listed the session again for every OCT acquisition; this reuses the list
    # >>>      it already has.
    def visit1(self, session, subject_label, acqs):
        """Same selection as the original handle_visit1_session, in the same order.

        `remaining` stands in for the original's fresh session.acquisitions() listing
        inside the loop: acquisitions leave it as they are moved.
        """
        if not any(a.files for a in acqs):
            return [], False
        remaining = list(acqs)
        found_oct = False
        to_move = []  # (acquisition, task or None)
        for acq in acqs:
            lab = acq.label.lower()
            if "oct" in lab:
                task = (OCT_PROTOCOL_LABEL, OCT_ASSIGNEE)
                oct_original = next(
                    (
                        a
                        for a in remaining
                        if "oct" in a.label.lower() and "original" in a.label.lower()
                    ),
                    None,
                )
                companion = next(
                    (
                        a
                        for a in remaining
                        if "oct" in a.label.lower()
                        and "original" not in a.label.lower()
                    ),
                    None,
                )
                if "original" in lab:
                    # the original moves with the first remaining non-original OCT; only the companion gets a task
                    found_oct = True
                    to_move.append((acq, None))
                    remaining.remove(acq)
                    if companion is not None:
                        to_move.append((companion, task))
                        remaining.remove(companion)
                elif not found_oct and not oct_original:
                    to_move.append((acq, task))  # lone OCT, no original in the session
                    remaining.remove(acq)
            elif "fp" in lab:
                to_move.append((acq, (FP_PROTOCOL_LABEL, FP_ASSIGNEE)))
                remaining.remove(acq)
        if not to_move:
            return [], False

        dqe = self.project(f"{SOURCE_GROUP}/DQE")
        dest_subject = self.dest.subject(dqe, subject_label)
        dest_session = self.dest.session(dest_subject, session.label)
        return [("acquisition", a, dest_session, a.label, t) for a, t in to_move], False

    def uploader_acknowledgement(self, session, project, acqs):
        if not any(a.files for a in acqs):
            return [], False
        review = self.project(f"{CERTIFICATION_GROUP}/Certification Review")
        dest_subject = self.dest.subject(
            review, f"ABID - {extract_prefix(project.label)}"
        )
        existing = self.dest.merge_target_or_move(session, dest_subject)
        if existing is None:
            return [], True  # moved whole
        print(f"Session '{session.label}' already exists, moving its acquisitions")
        return [("acquisition", a, existing, a.label, None) for a in acqs], False

    # >>> [F5] [C2] Same rules as handle_certification_session. The Certification Review
    # >>>           subject is created only when the session has files, and the initials
    # >>>           rename is sent together with the move.
    def certification(self, session, subject_label, project, acqs):
        with_files = [a for a in acqs if a.files]
        if not with_files:
            return [], False
        review = self.project(f"{CERTIFICATION_GROUP}/Certification Review")
        dest_subject = self.dest.subject(
            review, f"ABID - {extract_prefix(project.label)}"
        )
        dest_label = (
            "Equipment" if "equipment" in subject_label.lower() else subject_label
        )
        dest_session = self.dest.session(dest_subject, dest_label)
        steps = []
        for acq in with_files:
            label = acq.label
            initials = (extract_initials(f.name) for f in acq.files)
            initial = next((i for i in initials if i and i not in label), None)
            if initial:
                label = f"{label}_{initial}"
            steps.append(("acquisition", acq, dest_session, label, None))
        return steps, False

    def plan_session(self, session, project, acqs):
        """(steps, delete): the moves for one source session, and whether it ends up empty."""
        subject_label = session.subject.label if session.subject else "default"
        label = session.label.lower()
        moved_whole = False
        steps = []
        if session.label == "Patient Information":
            steps, moved_whole = self.patient_information(session, subject_label, acqs)
        elif "visit 1" in label:
            steps, moved_whole = self.visit1(session, subject_label, acqs)
        elif "uploader acknowledgement" in label:
            steps, moved_whole = self.uploader_acknowledgement(session, project, acqs)
        elif "certification" in label:
            steps, moved_whole = self.certification(
                session, subject_label, project, acqs
            )

        moved_ids = {s[1].id for s in steps if s[0] == "acquisition"}
        # >>> [C3] Your script reloaded every session and listed its acquisitions to
        # >>>      decide whether to delete it. The planner already knows what it read
        # >>>      and moved; the session is still checked once more right before it is
        # >>>      deleted.
        delete = not moved_whole and all(a.id in moved_ids for a in acqs)
        return steps, delete


# >>> [C7] [F1] Lists only the source group's projects. Your script listed every project
# >>>           on the site, filtered by group, and then listed the group's projects
# >>>           again for each one. Projects whose stats are still being computed are
# >>>           processed instead of stopping the script.
def source_projects(fw):
    projects = []
    for p in read(fw.get_group_projects, SOURCE_GROUP):
        if p.label in EXCLUDED_PROJECTS or not p.label[:1].isdigit():
            continue
        # stats are computed asynchronously; when missing, process the project anyway
        if p.stats and not p.stats.number_of.get("files"):
            continue
        projects.append(p)
    return projects


# >>> [C8] Reads a project's sessions 100 at a time, so each call stays small on large
# >>>      projects.
def project_sessions(fw, project_id):
    """All sessions of a project, PAGE_SIZE at a time (paged by id)."""
    after_id = None
    while True:
        kwargs = {"limit": PAGE_SIZE}
        if after_id:
            kwargs["after_id"] = after_id
        page = read(fw.get_project_sessions, project_id, **kwargs)
        yield from page
        if len(page) < PAGE_SIZE:
            return
        after_id = page[-1].id


def process_session(fw, fw_client, planner, project, session):
    if session.label == "Patient Information":
        # >>> [C8] Only Patient Information sessions need their info, so only those are
        # >>>      read in full, as in your script.
        session = read(fw.get_session, session.id)  # listings leave out info
    # >>> [C3] [C8] Each session's acquisitions are read once and reused for planning and
    # >>>           for the empty check.
    acqs = read(fw.get_session_acquisitions, session.id)
    steps, delete = planner.plan_session(session, project, acqs)
    dest = planner.dest
    for step in steps:  # moves first; the delete, if any, is always last
        if step[0] == "session":
            _, moving, dest_subject = step
            dest.move_session(moving, dest_subject)
        else:
            _, acq, dest_session, label, task = step
            dest.move_acquisition(acq, dest_session, label)
            if task:
                protocol_label, assignee = task
                create_task(
                    fw_client,
                    acq,
                    dest.full_session(dest_session),
                    protocol_label,
                    assignee,
                )
    if delete:
        # Like the original, check the session is empty right before deleting it,
        # so an upload that lands during the run is never deleted.
        # >>> [C3] The same final check as your script: the session is deleted only if it
        # >>>      is still empty right now, so a file uploaded during the run is never
        # >>>      deleted.
        if read(fw.get_session_acquisitions, session.id):
            print(
                f"Session '{session.label}' ({session.id}) is no longer empty, not deleting it"
            )
            return
        subject_label = session.subject.label if session.subject else "default"
        print(
            f"Session '{session.label}' of '{subject_label}' is now empty, deleting it"
        )
        fw.delete_session(
            session.id, delete_reason=ContainerDeleteReason.CONTAINER_IS_EMPTY
        )


def main():
    # >>> [F7] Prints the site, groups, worker count and where the key came from, and
    # >>>      warns if a reader-task value is empty.
    key, key_source = api_key()
    print(
        f"Site {key.split(':')[0]} (key from {key_source}); source group '{SOURCE_GROUP}', "
        f"certification group '{CERTIFICATION_GROUP}'; {WORKERS} worker(s)"
    )
    empty = [
        n
        for n in (
            "OCT_ASSIGNEE",
            "OCT_PROTOCOL_LABEL",
            "FP_ASSIGNEE",
            "FP_PROTOCOL_LABEL",
        )
        if not globals()[n]
    ]
    if empty:
        print(
            f"WARNING: {', '.join(empty)} empty at the top of this script; reader tasks will be created without them"
        )

    fw = flywheel.Client(key)
    fw_client = FWClient(api_key=key)
    planner = Planner(fw, Destinations(fw))
    projects = source_projects(fw)
    print("project_names:", [p.label for p in projects])

    failures = []
    # >>> [C9] Sessions are handed to WORKERS threads. Each session is read, planned and
    # >>>      moved by one thread, so its moves always finish before its delete.
    pool = ThreadPoolExecutor(max(1, WORKERS))
    try:
        for project in projects:
            futures = {
                pool.submit(process_session, fw, fw_client, planner, project, s): s
                for s in project_sessions(fw, project.id)
            }
            for future, session in futures.items():
                try:
                    future.result()
                # >>> [F6] One failed session is reported and skipped instead of stopping
                # >>>      the run. Failed sessions are listed at the end and the script
                # >>>      exits with an error code; running it again picks up whatever
                # >>>      is left.
                except Exception as e:  # keep going; report at the end
                    failures.append((project.label, session.label, session.id, repr(e)))
                    traceback.print_exc()
    # >>> [C9] Ctrl-C (or any error here) cancels the sessions that have not started and
    # >>>      lets the running ones finish, so none is left half-moved.
    except BaseException:
        # Ctrl-C, or an error in this thread: drop sessions that have not started and
        # let the running ones (at most WORKERS) finish, so none is left half-moved.
        print(
            "Stopping: cancelling sessions that have not started, waiting for the running ones"
        )
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    pool.shutdown()

    if failures:
        print(f"{len(failures)} session(s) failed; re-run the script to retry them:")
        for f in failures:
            print("  ", *f)
        sys.exit(1)


if __name__ == "__main__":
    main()

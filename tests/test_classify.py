from locium.classify import classify


def test_ordinary_reasoning_is_prose():
    text = (
        "The handler dispatches one message per batch. We decided to rename "
        "TrackingStatsCommunicator to EmsApiCommunicator because it was always "
        "a general EMS API client, never tracking-stats-only."
    )
    assert classify(text) is None


def test_prose_quoting_a_command_or_a_csv_row_stays_prose():
    text = (
        'To reproduce, run [Bash] docker compose up and check the row '
        '"11/26/2025","hacked","09/20/2025" in the export. The root cause was '
        "a stale cache entry that survived the deploy, which is why only some "
        "customers saw the error."
    )
    assert classify(text) is None


def test_a_pasted_csv_with_collapsed_newlines_is_data():
    # Exchange chunks lose their newlines at mine time; the seams remain.
    row = '"11/26/2025 03:10 PM","I have been hacked","09/20/2025",,22,21,437 '
    assert classify(row * 12) == "data"


def test_a_diff_hunk_is_data():
    diff = "\n".join(
        [
            "diff --git a/src/Service/EmsApiCommunicator.php b/src/Service/EmsApiCommunicator.php",
            "@@ -1,4 +1,4 @@",
            "-use App\\Service\\TrackingStatsCommunicator;",
            "+use App\\Service\\EmsApiCommunicator;",
            " class Controller",
            "+        private readonly EmsApiCommunicator $communicator,",
        ]
    )
    assert classify(diff) == "data"


def test_tool_result_lines_are_noise():
    dump = "\n".join(
        ["→ total 12"]
        + [f"→ -rw-rw-r-- 1 user user 3 apr 29 15:05 {i}_last_save" for i in range(8)]
        + ["one prose line about what this shows"]
    )
    assert classify(dump) == "noise"


def test_collapsed_tool_transcript_is_noise():
    text = (
        "[Bash] ls -la /docker/sign-api/apikeys/client/emsapi/ 2>&1 | head "
        "[Bash] ls -la /docker/sign-api/apikeys/client/support/ 2>&1 | head "
        "[Grep] EMS_API in config/ [Read /home/user/project/config/services.yaml]"
    )
    assert classify(text) == "noise"


def test_empty_text_is_prose():
    assert classify("   ") is None

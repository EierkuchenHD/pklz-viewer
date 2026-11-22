"""PKLZ Viewer: shows what is inside an audfprint .pklz database file.

    pythonw pklz_viewer.pyw <file.pklz>        open the window on that file
    pythonw pklz_viewer.pyw                    open the window and ask for a file
    python  pklz_viewer.pyw --check <file...>  no window: print one summary line per file
    python  pklz_viewer.pyw --hashes <file.pklz> <entry #|biggest> [picture.png]
                                               no window: one entry's hashes in figures,
                                               drawn into a PNG as well if one is named
    python  pklz_viewer.pyw --sources <file...> no window: every entry's source link
    python  pklz_viewer.pyw --register         add it to Explorer's "Open with" menu for .pklz (this user only)
    python  pklz_viewer.pyw --unregister       take it out again

"Show hashes" (or a double-click on an entry) draws the hashes of the selected
entry. Each hash is a pair of peaks in the song's spectrogram, so it is drawn as
a line from the first peak (time, frequency) to the second, coloured by how
often the same hash occurs anywhere in the table: the rare ones are what make
the entry findable. Under it, how many hashes each stretch of the entry has.
The picture follows the selection in the list; Up/Down in it moves on.

The Source column says where an entry came from when its name or the table's
name tells: a YouTube video (an id in the name), a WatZatSong post, a channel,
a Bandcamp search and so on. "Open source" or the right-click menu opens it.

Files are opened with WerZatSong's own loader, which only lets hash-table data
through: a .pklz is a pickle, and a plain pickle.load() of a downloaded file
could run code. Nothing is ever written to the file.
"""
import math
import os
import sys
import threading

sys.dont_write_bytecode = True
WZS_DIR = os.environ.get("WZS_DIR", r"C:\wzs\WerZatSong")
PROG_ID = "WerZatSong.PklzViewer"

# How audfprint turns a landmark into a hash and a time (libs/audfprint/audfprint_analyze.py):
# a 512-point FFT of the song at 11025 Hz with a 256-sample hop; the hash holds the first
# peak's frequency bin (8 bits), the step to the second peak in bins (6 bits, signed) and
# the gap between the two in frames (6 bits); the time stored with it is the first peak's frame.
N_FFT = 512
N_HOP = 256
DEFAULT_SR = 11025
F1_BITS, DF_BITS, DT_BITS = 8, 6, 6


def read_pklz(path):
    """ What a .pklz holds: a dict of summary values and the list of tracks. """
    lib = os.path.join(WZS_DIR, "libs", "audfprint")
    if not os.path.isfile(os.path.join(lib, "hash_table.py")):
        raise RuntimeError("WerZatSong's loader was not found in %s (set WZS_DIR)" % lib)
    if lib not in sys.path:
        sys.path.insert(0, lib)
    import contextlib, io
    import numpy as np
    import hash_table
    with contextlib.redirect_stdout(io.StringIO()):
        table = hash_table.HashTable(path)
    names = list(table.names)
    per_track = [int(x) for x in np.asarray(table.hashesperid).ravel()]
    per_track += [0] * (len(names) - len(per_track))
    counts = np.asarray(table.counts)
    depth = int(table.depth)
    buckets = int(counts.size)
    stored = int(np.minimum(counts, depth).sum())
    seen = int(counts.astype(np.int64).sum())
    tracks = [(i + 1, "" if n is None else str(n), per_track[i]) for i, n in enumerate(names)]
    loaded = dict(getattr(hash_table, "last_load", {}))
    info = {
        "File": path,
        "Size on disk": "%.1f MB" % (os.path.getsize(path) / 1e6),
        "Tracks": "{:,}".format(sum(1 for n in names if n)),
        "Hashes stored": "{:,}".format(stored),
        "Hashes dropped (bucket full)": "{:,} ({:.1f}%)".format(seen - stored, 100.0 * (seen - stored) / seen if seen else 0.0),
        "Buckets": "{:,} x {} entries, {:.1f}% of buckets full".format(buckets, depth, 100.0 * float((counts >= depth).sum()) / buckets),
        "Settings": ", ".join("%s=%s" % kv for kv in sorted(dict(table.params or {}).items())) or "(none)",
        "Format version": str(getattr(table, "ht_version", "?")),
        "Checksum": "verified" if loaded.get("verified") else "not verified this time",
    }
    table.__dict__.clear()
    return info, tracks


# ── Where an entry came from ────────────────────────────────────────────────
#
# A port of sourcelink.js, the rules that link WerZatSong's match results to their
# source; keep the two in step. They were worked out on a real corpus of matched
# files. Three tiers, and the label says which:
#   exact   the link lands on that track (an id survives in the name or path)
#   source  the channel or collection it came from (no per-track id survives)
#   search  the platform is known, the track is not: a scoped search
# No evidence, no link: an invented link that 404s is worse than none.

import re as _re

_CHUNK = r"(?:_part_?\d+|_split_\d+m-\d+m(?:\d+s)?|_trim_\d+|-\d+min_clips_\d+|_\d+-\d+m|-\d{3}|_\d+|\d{8})"
# yt-dlp's "[id]" at the end of a name, maybe followed by the part of a split recording.
_YOUTUBE_ID = _re.compile(r"\[([A-Za-z0-9_-]{11})\]\s*(?:\([^)]*\))?\s*" + _CHUNK + r"?\s*(?:\.[A-Za-z0-9]{2,5})?\Z")
# The fingerprinter's own downloads: "file-<title>-<id>.mp3", whole or in parts.
_FILE_PREFIXED = _re.compile(r"^file-.*-([A-Za-z0-9_-]{11})(?:_part\d+|_split_\d+m-\d+m(?:\d+s)?|_\d+-\d+m|-\d{3})?\.[A-Za-z0-9]{2,5}\Z")
# FPDbase (PotloodPKLZ) glues the id onto the title: "<Title><year><id>_<NNN>.mp3".
_GLUED_ID = _re.compile(r"([A-Za-z0-9_-]{11})_\d{3}\.[A-Za-z0-9]{2,5}\Z")
_WATZATSONG_ID = _re.compile(r"^WZS(\d+)(?:-[a-z]+)?\.[A-Za-z0-9]+\Z", _re.I)
_WATZATSONG_DOWNLOAD = _re.compile(r"^wzs-(\d{1,12}) - ", _re.I)
# "<id> - <title>": weak on its own (usually an artist), so it needs more evidence.
_LEADING_ID = _re.compile(r"^(?!file-)([A-Za-z0-9_-]{11}) - ", _re.I)
_WORDS_NOT_ID = _re.compile(r"^(?:[A-Za-z]?\d{1,3}|[a-z]+|[A-Z][a-z]+)[-_.]?[A-Z]?[a-z]+\Z")
_ACIDPLANET_ITEM = _re.compile(r"(acidplanet-audio-\d{6,})", _re.I)
_AUDIOJUNGLE = _re.compile(r"[?&](\d{5,})_([a-z0-9-]+?)_by_[A-Za-z0-9_-]+\Z", _re.I)
# yt-dlp names a Mixcloud show "<title> [<user>_<slug>]".
_MIXCLOUD_ID = _re.compile(r"\[([A-Za-z0-9_-]+)_((?:[^\W_]|-)+)\]")
_SOUNDCLOUD_FOLDER = _re.compile(r"(?:^|/)soundcloud-(\d{6,12})(?:/|\Z)", _re.I)
_SOUNDCLOUD_USER = _re.compile(r"(?:^|/)soundcloud--([A-Za-z0-9_-]+)(?:/|\Z)", _re.I)
_NUMERIC_BRACKET = _re.compile(r"\[(\d{6,12})\]")
_NUMERIC_NAME = _re.compile(r"^(\d{4,})\.[A-Za-z0-9]+\Z")
_SOUNDCLICK_ID = _re.compile(r"ID\[(\d+)\]")
_CHANNEL_ID = _re.compile(r"(UC[A-Za-z0-9_-]{22})(?=\Z|[^A-Za-z0-9_-]|[-_]\d)")
# A table named for a channel handle: "@mauropiffero1-4.pklz" (only the shard number comes off).
_HANDLE_SHARD = _re.compile(r"^(@[A-Za-z0-9_.-]+?)(?:-\d+)?\Z")
_WAYBACK = _re.compile(r"web\.archive\.org/web/([0-9]{4,14}(?:[a-z_]{2,3})?)/(https?)_/(.+)\Z", _re.I)
_IA_ITEM = _re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]{3,})/")
_YTDLP_FORMAT = _re.compile(r"\((?:\d+kbit_[A-Za-z0-9]+|\d+k)\)")
# yt-dlp's default template "<title>-<id>": only trusted with more evidence.
_YOUTUBE_TRAILING_ID = _re.compile(r"-([A-Za-z0-9_-]{11})(?:_\d+)?\.[A-Za-z0-9]{2,5}\Z")
_YTDLP_DUMP = _re.compile(r"(?:^|/)yt-dlp(?:\Z|/)", _re.I)
# The lookalikes yt-dlp writes for characters Windows forbids in a file name.
_YTDLP_SUBSTITUTE = _re.compile("[\u29f8\uff02\uff1f\uff5c\uff1a\uff0a\uff1c\uff1e]")
_BLOGSPOT_DOMAIN = _re.compile(r"([a-z0-9][a-z0-9-]*)\.(blogspot|eklablog)\.com", _re.I)
_BLOGSPOT_FOLDER = _re.compile(r"^(.+?)[ _-]blogspot\Z", _re.I)
_ARCHIVE_ORG_FOLDER = _re.compile(r"(?:^|/)archive\.org/(?:archiveorg--)?([^/]{4,})/", _re.I)
_FMA_NAME = _re.compile(r"^(.+?)_-_")
_GENERIC_FOLDER = _re.compile(r"^(?:various|va|disc\s*\d*|cd\s*\d*|mp3s?|audio|tracks?|untitled)\Z", _re.I)


def _norm(value):
    return str(value or "").strip()


def _uri(value):
    """ JavaScript's encodeURIComponent. """
    from urllib.parse import quote
    return quote(value, safe="!~*'()")


def _word_like(token):
    return (bool(_re.fullmatch(r"[A-Za-z]+", token)) and len(_re.findall(r"[aeiouy]", token, _re.I)) >= 3
            and not _re.search(r"[^aeiouy]{5}", token, _re.I))


def _bracket_id_ok(token):
    return bool(_re.search(r"[0-9_-]", token)) or bool(_re.search(r"[A-Z]", token[1:])) or not _word_like(token)


def _id_like(token):
    return bool(_re.search(r"[0-9_-]", token) and _re.search(r"[A-Z]", token) and _re.search(r"[a-z]", token))


def _id_like_loose(token):
    return _id_like(token) or (bool(_re.search(r"[a-z]", token)) and len(_re.findall(r"[A-Z]", token[1:])) >= 2)


def _letters(value):
    import unicodedata
    return "".join(ch for ch in unicodedata.normalize("NFKC", _norm(value)).lower() if ch.isalnum())


def _shard_family(pklz):
    family = _re.sub(r"\.pklz\Z", "", _norm(pklz), flags=_re.I)
    return _re.sub(r"[-_ ]?#?\d+(?:[-_]\d+)*\Z", "", family).strip()


def _starts(value, prefix):
    return _norm(value).lower().startswith(prefix.lower())


def _searchable_title(matched):
    title = _re.sub(r"\.[A-Za-z0-9]{2,5}\Z", "", _norm(matched), count=1)
    title = _re.sub(r"\[[A-Za-z0-9_-]{11}\]", "", title)
    title = _re.sub(r"\((?:\d+kbit_[A-Za-z0-9]+|\d+k)\)", "", title)
    title = _re.sub(r"_\d+\Z", "", title, count=1)
    for lookalike, real in (("\uff02", '"'), ("\uff1f", "?"), ("\uff5c", "|"), ("\u29f8", "/"), ("\u2044", "/")):
        title = title.replace(lookalike, real)
    return _re.sub(r"\s+", " ", title).strip()


def _link(url, kind, tier, label):
    return {"url": url, "kind": kind, "tier": tier, "label": label}


def source_link_of(pklz, matched, matched_path):
    """ sourceLink(pklz, matched, matchedPath) of sourcelink.js: pklz is the table's file
        name, matched the entry's file name, matched_path the name as the table stores it. """
    name = _norm(matched)
    path = _norm(matched_path).replace("\\", "/")
    shard = _norm(pklz)
    family = _shard_family(shard)
    if not name and not path:
        return None

    def youtube(video):
        return _link("https://youtu.be/%s" % video, "youtube", "exact", "YouTube")

    # exact: an id that names this specific track
    m = _YOUTUBE_ID.search(name)
    if m and _bracket_id_ok(m.group(1)):
        return youtube(m.group(1))
    m = _FILE_PREFIXED.search(name)
    if m and _bracket_id_ok(m.group(1)):
        return youtube(m.group(1))
    m = _WATZATSONG_ID.search(name) or _WATZATSONG_DOWNLOAD.search(name)
    if m:
        return _link("https://www.watzatsong.com/en/name-that-tune/%s.html" % m.group(1), "watzatsong", "exact", "WatZatSong")
    m = _LEADING_ID.search(name)
    if m and _YTDLP_SUBSTITUTE.search(name) and _re.search(r"\d", m.group(1)) and _id_like(m.group(1)) and not _WORDS_NOT_ID.search(m.group(1)):
        return youtube(m.group(1))
    m = _SOUNDCLICK_ID.search(name)
    if m and (_starts(family, "soundclick") or _re.search("soundclick", path, _re.I)):
        return _link("https://www.soundclick.com/music/songInfo.cfm?songID=%s" % m.group(1), "soundclick", "exact", "SoundClick")
    m = _NUMERIC_NAME.search(name)
    if m and (_starts(family, "reverbnation") or _re.search("reverbnation", path, _re.I)):
        return _link("https://legacy.reverbnation.com/open_graph/song/%s" % m.group(1), "reverbnation", "exact", "ReverbNation")
    m = _WAYBACK.search(path)
    if m:
        return _link("https://web.archive.org/web/%s/%s://%s" % (m.group(1), m.group(2).lower(), m.group(3)), "wayback", "exact", "Wayback Machine")
    m = (_AUDIOJUNGLE.search(path) or _AUDIOJUNGLE.search(name)) if _re.search(r"envato\.com|audiojungle", "%s %s" % (path, shard), _re.I) else None
    if m:
        return _link("https://audiojungle.net/item/%s/%s" % (m.group(2).lower(), m.group(1)), "audiojungle", "exact", "AudioJungle")
    m = _ACIDPLANET_ITEM.search(path) or _ACIDPLANET_ITEM.search(name)
    if m:
        return _link("https://archive.org/details/%s" % m.group(1).lower(), "archive", "exact", "Internet Archive")
    mixes = list(_MIXCLOUD_ID.finditer(name))
    if mixes:
        mix = mixes[-1]
        title = _re.sub(r"^(?:5m_Clips-|mixcloud_)", "", name[:mix.start()], flags=_re.I)
        spelled = _letters(title) != "" and _letters(title) == _letters(mix.group(2))
        if spelled or _re.search(r"mixcloud|(?:^|/)mixc-", path, _re.I):
            return _link("https://www.mixcloud.com/%s/%s/" % (mix.group(1), _uri(mix.group(2))), "mixcloud", "exact", "Mixcloud")
    folders = path[:path.rfind("/")] if "/" in path else ""
    sc_folder = _SOUNDCLOUD_FOLDER.search(folders)
    where = "%s %s" % (folders, shard)
    sc_bracket = _NUMERIC_BRACKET.search(name) if _re.search("soundcloud", where, _re.I) and not _re.search("bandcamp", where, _re.I) else None
    sc_id = (sc_folder and sc_folder.group(1)) or (sc_bracket and sc_bracket.group(1))
    if sc_id:
        return _link("https://w.soundcloud.com/player/?url=https%%3A//api.soundcloud.com/tracks/%s" % sc_id, "soundcloud", "exact", "SoundCloud")
    if _starts(family, "fpdbase") or _re.search("potloodpklz", path, _re.I):
        m = _GLUED_ID.search(name)
        if m and _bracket_id_ok(m.group(1)):
            return youtube(m.group(1))
    m = _YOUTUBE_TRAILING_ID.search(name)
    if m:
        if _YTDLP_DUMP.search(path) and _id_like_loose(m.group(1)):
            return youtube(m.group(1))
        if _YTDLP_SUBSTITUTE.search(name) and _id_like(m.group(1)):
            return youtube(m.group(1))

    # source: the channel or collection, when no track id survives
    if _starts(family, "ia-") and path and not _re.match(r"[A-Za-z]:/", path) and not path.startswith("/"):
        m = _IA_ITEM.search(path)
        if m:
            return _link("https://archive.org/details/%s" % m.group(1), "archive", "source", "Internet Archive")
    if _re.search(r"myspace[_ ]?dragon[_ ]?hoard", shard, _re.I) or _re.search("myspace_dragon_hoard", path, _re.I):
        return _link("https://archive.org/details/myspace_dragon_hoard_2010", "myspace", "source", "MySpace Dragon Hoard")
    m = _CHANNEL_ID.search(path) or _CHANNEL_ID.search(shard)
    if m:
        return _link("https://www.youtube.com/channel/%s" % m.group(1), "youtube-channel", "source", "YouTube channel")
    m = _HANDLE_SHARD.search(_re.sub(r"\.pklz\Z", "", shard, flags=_re.I))
    if m:
        return _link("https://www.youtube.com/%s" % m.group(1), "youtube-channel", "source", "YouTube channel")
    m = _SOUNDCLOUD_USER.search(path)
    if m:
        return _link("https://soundcloud.com/%s" % m.group(1), "soundcloud-user", "source", "SoundCloud")

    # search: platform known, track not
    title = _searchable_title(name)
    segments = [s for s in path.split("/") if s]
    stated = next((s for s in segments if _BLOGSPOT_DOMAIN.search(s)), None)
    named = next((s for s in segments if _BLOGSPOT_FOLDER.search(s)), None)
    blog = stated or named
    if blog:
        domain = _BLOGSPOT_DOMAIN.search(blog)
        folder = _BLOGSPOT_FOLDER.search(blog)
        host = ("%s.%s.com" % (domain.group(1).lower(), domain.group(2).lower()) if domain
                else "%s.blogspot.com" % _re.sub(r"[^a-z0-9-]+", "", folder.group(1).lower()))
        raw_parent = segments[-2] if len(segments) > 1 else ""
        parent = _re.sub(r"[\[(]\s*[a-z0-9-]+\.(?:blogspot|eklablog)\.com\s*[\])]", "", raw_parent, flags=_re.I)
        parent = _re.sub(r"[a-z0-9-]+\.(?:blogspot|eklablog)\.com", "", parent, flags=_re.I)
        parent = _re.sub(r"\s{2,}", " ", parent).strip()
        use_parent = bool(parent) and parent != named and len(parent) >= 4 and not _GENERIC_FOLDER.search(parent)
        needle = parent if use_parent else title
        if host and needle:
            return _link("https://%s/search?q=%s" % (host, _uri(needle)), "blogspot", "search", "blog search")
    m = _ARCHIVE_ORG_FOLDER.search(path)
    if m:
        words = _re.sub(r"\s+", " ", _re.sub(r"[_-]+", " ", m.group(1))).strip()
        if words:
            return _link("https://archive.org/search?query=%s" % _uri(words), "archive-search", "search", "Internet Archive search")
    if not title:
        return None
    if _starts(family, "freemusicarchive") or _re.search("freemusicarchive", path, _re.I):
        m = _FMA_NAME.search(name)
        artist = _re.sub(r"\s+", " ", (m.group(1) if m else title).replace("_", " ")).strip()
        if artist:
            return _link("https://freemusicarchive.org/search?quicksearch=%s" % _uri(artist), "fma-search", "search", "Free Music Archive search")
    if _starts(family, "bandcamp") or _re.search("bandcamp", path, _re.I):
        return _link("https://bandcamp.com/search?q=%s" % _uri(title), "bandcamp", "search", "Bandcamp search")
    if _YTDLP_FORMAT.search(name):
        return _link("https://www.youtube.com/results?search_query=%s" % _uri(title), "youtube-search", "search", "YouTube search")
    return None


def source_link(table, entry):
    """ Where entry <entry> of table <table> (a path or a file name) came from, or None. """
    if not entry:
        return None
    stored = str(entry)
    return source_link_of(os.path.basename(str(table or "")), _re.split(r"[\\/]", stored)[-1], stored)


TIER_NOTE = {"exact": "", "source": " (where it came from, not the track itself)", "search": " (a search: no id survives)"}


# ── The hashes of each entry ────────────────────────────────────────────────

class HashIndex(object):
    """ Every hash stored in one table, by entry. Built once per file (a few seconds for
        a large table); after that any entry's hashes come out in milliseconds. The table
        itself (about 400 MB in memory) is let go once this is built. """

    def __init__(self, path):
        lib = os.path.join(WZS_DIR, "libs", "audfprint")
        if not os.path.isfile(os.path.join(lib, "hash_table.py")):
            raise RuntimeError("WerZatSong's loader was not found in %s (set WZS_DIR)" % lib)
        if lib not in sys.path:
            sys.path.insert(0, lib)
        import contextlib, io
        import numpy as np
        import hash_table
        with contextlib.redirect_stdout(io.StringIO()):
            table = hash_table.HashTable(path)
        try:
            depth = int(table.depth)
            timebits = int(table.maxtimebits)
            flat = np.asarray(table.table).reshape(-1)
            # The table stores (entry + 1) << timebits | time, so an empty slot is 0.
            where = np.flatnonzero(flat)
            values = flat[where]
            counts = np.asarray(table.counts).astype(np.int64)
            buckets = (where // depth).astype(np.int32)
            # Slots past a bucket's count are not entries (the format allows them; none were seen).
            live = (where % depth) < np.minimum(counts, depth)[buckets]
            self.ids = ((values >> np.uint32(timebits)).astype(np.int64) - 1).astype(np.int32)[live]
            self.times = (values & np.uint32((1 << timebits) - 1)).astype(np.int32)[live]
            self.hashes = buckets[live]
            # How many hashes went into each bucket, every entry's together (more than the
            # depth means the bucket is full and some of them were dropped).
            self.counts = counts
            self.depth = depth
            self.timebits = timebits
            self.hashbits = int(table.hashbits)
            self.names = list(table.names)
            self.made = [int(x) for x in np.asarray(table.hashesperid).ravel()]
            sr = float(dict(table.params or {}).get("samplerate") or DEFAULT_SR)
            self.seconds_per_frame = N_HOP / sr
            self.hz_per_bin = sr / N_FFT
            # Times are kept in <timebits> bits: 16,384 frames is 6:20.4 at the usual settings.
            self.wrap_seconds = (1 << timebits) * self.seconds_per_frame
            self.path = path
        finally:
            table.__dict__.clear()

    def entry(self, k):
        """ (frame times, hashes) of entry k (0-based, the list's # minus one), in time order. """
        import numpy as np
        pick = np.flatnonzero(self.ids == k)
        times, hashes = self.times[pick], self.hashes[pick]
        order = np.argsort(times, kind="stable")
        return times[order], hashes[order]

    def biggest(self):
        """ The entry with the most hashes kept (0-based). """
        import numpy as np
        if not self.ids.size:
            return 0
        return int(np.bincount(self.ids, minlength=len(self.names)).argmax())


def landmarks(index, times, hashes):
    """ The peak pairs behind an entry's hashes, in seconds and Hz, with how often each
        hash occurs in the whole table and the colour class that follows from it. """
    import numpy as np
    dt = hashes & ((1 << DT_BITS) - 1)
    df = (hashes >> DT_BITS) & ((1 << DF_BITS) - 1)
    df = np.where(df >= (1 << (DF_BITS - 1)), df - (1 << DF_BITS), df)
    f1 = (hashes >> (DF_BITS + DT_BITS)) & ((1 << F1_BITS) - 1)
    count = index.counts[hashes]
    shade = np.where(count >= index.depth, 3, np.where(count <= 2, 0, np.where(count <= 10, 1, 2))).astype(np.int8)
    return {
        "t": times * index.seconds_per_frame,
        "dt": dt * index.seconds_per_frame,
        "f1": f1 * index.hz_per_bin,
        "f2": (f1 + df) * index.hz_per_bin,
        "hash": hashes,
        "count": count,
        "shade": shade,
    }


def entry_facts(index, k, times, hashes):
    import numpy as np
    facts = {"made": index.made[k] if k < len(index.made) else 0, "kept": int(times.size)}
    if times.size:
        seconds = times * index.seconds_per_frame
        count = index.counts[hashes]
        facts["first"] = float(seconds.min())
        facts["last"] = float(seconds.max())
        facts["per_second"] = times.size / max(facts["last"] - facts["first"], index.seconds_per_frame)
        facts["distinct"] = int(np.unique(hashes).size)
        facts["full"] = float((count >= index.depth).mean())
        facts["rare"] = float((count <= 2).mean())
        facts["wrapped"] = facts["last"] > 0.97 * index.wrap_seconds
    return facts


def clock(seconds, decimals=1):
    """ Seconds as m:ss (h:mm:ss from an hour on) with <decimals> places: 75.3 -> "1:15.3". """
    unit = 10 ** decimals
    whole, part = divmod(int(round(max(0.0, float(seconds)) * unit)), unit)
    minutes, secs = divmod(whole, 60)
    hours, minutes = divmod(minutes, 60)
    text = "%d:%02d:%02d" % (hours, minutes, secs) if hours else "%d:%02d" % (minutes, secs)
    return text + (".%0*d" % (decimals, part) if decimals else "")


def facts_text(index, facts):
    made, kept = facts["made"], facts["kept"]
    if not kept:
        return "None of this entry's hashes are in the table%s." % (" ({:,} were made)".format(made) if made else "")
    if made >= kept:
        first = "{:,} of {:,} hashes kept ({:.0f}%)".format(kept, made, 100.0 * kept / made if made else 100.0)
        if made > kept:
            first += ", {:,} lost to full buckets".format(made - kept)
    else:
        first = "{:,} hashes kept".format(kept)
    text = " · ".join([
        first,
        "%s to %s" % (clock(facts["first"]), clock(facts["last"])),
        "%.0f per second" % facts["per_second"],
        "{:,} different hashes".format(facts["distinct"]),
        "%.0f%% rare (in the whole table at most twice)" % (100.0 * facts["rare"]),
        "%.1f%% in full buckets" % (100.0 * facts["full"]),
    ])
    if facts["wrapped"]:
        text += ("\nThis table keeps times modulo %s (%s frames) and this entry reaches that point: "
                 "anything longer is folded back onto the start of the picture." % (clock(index.wrap_seconds), "{:,}".format(1 << index.timebits)))
    if index.hashbits != F1_BITS + DF_BITS + DT_BITS:
        text += "\nThis table uses %d-bit hashes, so the frequencies drawn are only approximate." % index.hashbits
    return text


# ── Drawing ─────────────────────────────────────────────────────────────────

BG = (13, 17, 23)
PLOT_BG = (19, 24, 31)
GRID = (42, 50, 62)
TEXT = (210, 216, 222)
DIM = (130, 140, 152)
BARS = (88, 166, 255)
# By how many hashes went into the same bucket of the table, this entry's included:
# (peak colour, line colour). The rare ones are drawn last, on top.
SHADES = [
    ((255, 214, 102), (196, 156, 62)),   # 1-2: as good as this entry's own
    ((255, 140, 66), (178, 96, 46)),     # 3-10
    ((88, 166, 255), (52, 104, 166)),    # 11 up to a full bucket
    ((86, 100, 122), (58, 68, 84)),      # bucket full: some hashes went missing there
]
# More peak pairs than this on screen: the first peaks only, no lines (keeps zooming quick).
LINES_MAX = 15000


def _font(size):
    from PIL import ImageFont
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _time_step(span, pixels, min_px):
    for step in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600):
        if span / step * min_px <= pixels:
            return step
    return 7200


def full_view(index, lm):
    """ The whole entry: from 0 to the end of its last peak pair, with a little room. """
    if not len(lm["t"]):
        return (0.0, 10.0)
    return (0.0, max(float((lm["t"] + lm["dt"]).max()) * 1.02, 1.0))


def draw_entry(index, lm, size, view=None, header=None, scale=1.0):
    """ The picture of one entry's hashes, as a PIL image of <size> pixels.

        view: (first, last) second shown; the whole entry if None. header: lines of text
        above the picture (for a saved PNG; the window has labels of its own).
        Returns (image, geometry); geometry maps pixels back to times, frequencies and
        peak pairs for the pointer read-out (None if the picture is too small). """
    import numpy as np
    from PIL import Image, ImageDraw

    s = float(scale)
    width, height = int(size[0]), int(size[1])
    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)
    font = _font(int(round(12 * s)))
    small = _font(int(round(11 * s)))
    t0, t1 = view or full_view(index, lm)
    span = max(t1 - t0, 1e-6)
    fmax = (1 << F1_BITS) * index.hz_per_bin

    y = int(8 * s)
    for line in header or []:
        draw.text((int(12 * s), y), line, font=font, fill=TEXT)
        y += int(19 * s)

    legend_y = y
    x0, x1 = int(54 * s), width - int(14 * s)
    py0 = legend_y + int(24 * s)
    sy1 = height - int(22 * s)
    sy0 = sy1 - max(int(46 * s), int(height * 0.16))
    py1 = sy0 - int(8 * s)
    if x1 - x0 < 60 or py1 - py0 < 40:
        draw.text((10, 10), "Make the window bigger to see the hashes.", font=font, fill=TEXT)
        return img, None
    pw, ph, sh = x1 - x0, py1 - py0, sy1 - sy0

    # What the colours mean.
    lx = float(x0)
    label = "Same hash in the whole table:"
    draw.text((lx, legend_y + 2 * s), label, font=small, fill=DIM)
    lx += draw.textlength(label, font=small) + 10 * s
    for (dot, _), label in zip(SHADES, ["1-2 times", "3-10", "11-%d" % (index.depth - 1), "%d+ (bucket full)" % index.depth]):
        draw.rectangle([lx, legend_y + 6 * s, lx + 9 * s, legend_y + 15 * s], fill=dot)
        lx += 14 * s
        draw.text((lx, legend_y + 2 * s), label, font=small, fill=TEXT)
        lx += draw.textlength(label, font=small) + 14 * s

    plot = Image.new("RGB", (pw, ph), PLOT_BG)
    pdraw = ImageDraw.Draw(plot)
    strip = Image.new("RGB", (pw, sh), PLOT_BG)
    sdraw = ImageDraw.Draw(strip)

    def px(t):
        return (t - t0) / span * pw

    def py(f):
        return ph - f / fmax * ph

    # Grid: time across, kilohertz up.
    step = _time_step(span, pw, 84 * s)
    decimals = 0 if step >= 1 else (1 if step >= 0.1 else 2)
    first = math.ceil(t0 / step - 1e-9)
    for n in range(first, first + 400):
        tick = n * step
        if tick > t1 + 1e-9:
            break
        x = px(tick)
        pdraw.line([(x, 0), (x, ph)], fill=GRID)
        sdraw.line([(x, 0), (x, sh)], fill=GRID)
        label = clock(tick, decimals)
        draw.text((x0 + x - draw.textlength(label, font=small) / 2, sy1 + 4 * s), label, font=small, fill=DIM)
    khz = 1000 if ph >= 110 * s else 2000
    for f in range(0, int(fmax) + 1, khz):
        yy = py(f)
        pdraw.line([(0, yy), (pw, yy)], fill=GRID)
        label = "%d kHz" % (f // 1000)
        draw.text((x0 - 7 * s - draw.textlength(label, font=small), py0 + yy - 8 * s), label, font=small, fill=DIM)

    # The peak pairs.
    t, dt, f1, f2, shade = lm["t"], lm["dt"], lm["f1"], lm["f2"], lm["shade"]
    seen = np.flatnonzero((t <= t1) & (t + dt >= t0))
    xs = px(t[seen])
    ys = py(np.clip(f1[seen], 0, fmax))
    lines = seen.size <= LINES_MAX
    if lines:
        xe = px((t + dt)[seen])
        ye = py(np.clip(f2[seen], 0, fmax))
        thick = max(1, int(round(s)))
        r = 1.7 * s
        for c in (3, 2, 1, 0):
            mine = np.flatnonzero(shade[seen] == c)
            dot, colour = SHADES[c]
            for i in mine:
                pdraw.line([(xs[i], ys[i]), (xe[i], ye[i])], fill=colour, width=thick)
            for i in mine:
                pdraw.ellipse([xs[i] - r, ys[i] - r, xs[i] + r, ys[i] + r], fill=dot)
    else:
        pixels = np.array(plot)
        dot_px = max(1, int(round(1.5 * s)))
        for c in (3, 2, 1, 0):
            mine = np.flatnonzero(shade[seen] == c)
            xi = np.clip(xs[mine].astype(np.int64), 0, pw - 1)
            yi = np.clip(ys[mine].astype(np.int64), 0, ph - 1)
            for ox in range(dot_px):
                for oy in range(dot_px):
                    pixels[np.minimum(yi + oy, ph - 1), np.minimum(xi + ox, pw - 1)] = SHADES[c][0]
        plot = Image.fromarray(pixels)
        pdraw = ImageDraw.Draw(plot)
        note = "%s peak pairs on screen: first peaks only - zoom in for the pairs" % "{:,}".format(seen.size)
        note_x = pw - pdraw.textlength(note, font=small) - 10 * s
        pdraw.rectangle([note_x - 6 * s, 3 * s, pw - 4 * s, 19 * s], fill=PLOT_BG)
        pdraw.text((note_x, 4 * s), note, font=small, fill=DIM)
    if not t.size:
        pdraw.text((12 * s, 10 * s), "No hashes of this entry are stored in this table.", font=font, fill=TEXT)

    # How many hashes each stretch has, per second: about 60 stretches across, at least 0.1 s each.
    width_s = next((w for w in (0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300) if span / w <= 60), 600)
    edge0 = math.floor(t0 / width_s) * width_s
    bins = max(1, int(math.ceil((t1 - edge0) / width_s)))
    inside = t[(t >= edge0) & (t < edge0 + bins * width_s)]
    counted = np.histogram(inside, bins=bins, range=(edge0, edge0 + bins * width_s))[0] if inside.size else np.zeros(bins)
    rate = counted / width_s
    top = max(float(rate.max()), 1.0)
    for i in np.flatnonzero(rate > 0):
        h = rate[i] / top * (sh - 5 * s)
        left_x, right_x = px(edge0 + i * width_s), px(edge0 + (i + 1) * width_s)
        sdraw.rectangle([left_x + 1, sh - h, max(right_x - 1, left_x + 1), sh], fill=BARS)
    label = "hashes per second (each bar %s s)" % ("%g" % width_s)
    sdraw.rectangle([3 * s, 2 * s, 12 * s + sdraw.textlength(label, font=small), 17 * s], fill=PLOT_BG)
    sdraw.text((7 * s, 3 * s), label, font=small, fill=DIM)
    label = "%.0f/s" % top
    draw.text((x0 - 7 * s - draw.textlength(label, font=small), sy0 - 1 * s), label, font=small, fill=DIM)
    draw.text((x0 - 7 * s - draw.textlength("0", font=small), sy1 - 14 * s), "0", font=small, fill=DIM)

    img.paste(plot, (x0, py0))
    img.paste(strip, (x0, sy0))
    geometry = {
        "x0": x0, "x1": x1, "py0": py0, "py1": py1, "sy0": sy0, "sy1": sy1,
        "t0": t0, "t1": t1, "fmax": fmax, "lines": lines,
        "xs": xs + x0, "ys": ys + py0, "idx": seen,
    }
    return img, geometry


def write_hashes_csv(lm, target):
    """ One row per hash of the entry: when, the two peaks, the gap, the hash, how often it is in the table. """
    import csv
    with open(target, "w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(["time_s", "first_peak_hz", "second_peak_hz", "gap_ms", "hash", "times_in_table"])
        for i in range(lm["t"].size):
            writer.writerow(["%.3f" % lm["t"][i], "%.1f" % lm["f1"][i], "%.1f" % lm["f2"][i], "%.0f" % (lm["dt"][i] * 1000.0),
                             "%05X" % int(lm["hash"][i]), int(lm["count"][i])])


def _safe_stem(name, fallback):
    import re
    stem = os.path.splitext(os.path.basename(str(name or "").replace("\\", "/")))[0] or fallback
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", stem)[:90]


class HashWindow(object):
    """ "Show hashes": one entry's hashes drawn; follows the selection in the list. """

    def __init__(self, root, need_index, step, on_close):
        import tkinter as tk
        from tkinter import ttk
        self.need_index = need_index    # need_index(callback): callback(index, error) once the file's hashes are read
        self.step = step                # step(+1 / -1): select the next / previous entry in the list
        self.on_close = on_close
        self.entry = None               # (k, name)
        self.index = None
        self.waiting = False
        self.lm = None
        self.facts = None
        self.view = None
        self.geometry = None
        self.photo = None
        self.soon = None
        self.drag = None
        self.drawn = 0                  # pictures drawn, for the self-test

        top = self.top = tk.Toplevel(root)
        top.title("Hashes")
        top.geometry("1180x700")
        top.protocol("WM_DELETE_WINDOW", self.close)
        head = ttk.Frame(top, padding=(10, 8, 10, 0))
        head.pack(fill="x")
        self.title_label = ttk.Label(head, text="", font=("Segoe UI", 11, "bold"))
        self.title_label.pack(anchor="w")
        self.link = None
        self.source_label = ttk.Label(head, text="", foreground="#0b5cad", cursor="hand2")
        self.source_label.pack(anchor="w")
        self.source_label.bind("<Button-1>", lambda e: self.open_source())
        self.facts_label = ttk.Label(head, text="", justify="left", wraplength=1150)
        self.facts_label.pack(anchor="w", pady=(3, 0))
        bar = ttk.Frame(top, padding=(10, 6))
        bar.pack(fill="x")
        ttk.Button(bar, text="Whole entry", command=self.fit).pack(side="left")
        ttk.Button(bar, text="Save picture...", command=self.save_picture).pack(side="left", padx=6)
        ttk.Button(bar, text="Save hashes as CSV...", command=self.save_csv).pack(side="left")
        ttk.Label(bar, text="Wheel: zoom    Drag: move    Double-click: whole entry    Up/Down: next entry",
                  foreground="#666666").pack(side="right")
        self.canvas = tk.Canvas(top, background="#0d1117", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=10)
        self.image_item = self.canvas.create_image(0, 0, anchor="nw")
        self.message_item = self.canvas.create_text(24, 40, anchor="nw", fill="#d2d8de", font=("Segoe UI", 11), text="")
        self.status = ttk.Label(top, text=" ", padding=(10, 4, 10, 8))
        self.status.pack(fill="x")

        c = self.canvas
        c.bind("<Configure>", lambda e: self.redraw_soon())
        c.bind("<ButtonPress-1>", self.press)
        c.bind("<B1-Motion>", self.move)
        c.bind("<ButtonRelease-1>", lambda e: setattr(self, "drag", None))
        c.bind("<Double-Button-1>", lambda e: self.fit())
        c.bind("<Motion>", self.point)
        c.bind("<Leave>", lambda e: self.status.configure(text=" "))
        top.bind("<MouseWheel>", self.wheel)
        top.bind("<Up>", lambda e: self.step(-1))
        top.bind("<Down>", lambda e: self.step(1))
        top.bind("<Home>", lambda e: self.fit())
        for key in ("<plus>", "<KP_Add>", "<equal>"):
            top.bind(key, lambda e: self.zoom(0.7))
        for key in ("<minus>", "<KP_Subtract>"):
            top.bind(key, lambda e: self.zoom(1.0 / 0.7))
        top.bind("<Escape>", lambda e: self.close())

    # entries

    def show(self, k, name, link=None):
        self.entry = (k, name)
        label = name or "(removed entry)"
        self.top.title("Hashes - %s" % os.path.basename(label.replace("\\", "/")))
        self.title_label.configure(text="%s   ·   entry #%s" % (label, "{:,}".format(k + 1)))
        self.link = link
        self.source_label.configure(text="Source: %s%s   %s" % (link["label"], TIER_NOTE.get(link["tier"], ""), link["url"]) if link else "")
        if self.index is None:
            self.facts_label.configure(text="")
            self.message("Reading the hashes of every entry in this table (once per file)...")
            if not self.waiting:
                self.waiting = True
                self.need_index(self.got_index)
            return
        self.load_entry()

    def got_index(self, index, error):
        self.waiting = False
        if error is not None:
            self.message("Could not read the hashes:\n%s" % error)
            return
        self.index = index
        if self.entry is not None:
            self.load_entry()

    def load_entry(self):
        k, _ = self.entry
        times, hashes = self.index.entry(k)
        self.lm = landmarks(self.index, times, hashes)
        self.facts = entry_facts(self.index, k, times, hashes)
        self.facts_label.configure(text=facts_text(self.index, self.facts))
        self.view = full_view(self.index, self.lm)
        self.message("")
        self.redraw()

    # drawing

    def message(self, text):
        self.canvas.itemconfigure(self.message_item, text=text)
        self.canvas.tag_raise(self.message_item)

    def redraw_soon(self):
        if self.soon is not None:
            self.top.after_cancel(self.soon)
        self.soon = self.top.after(60, self.redraw)

    def redraw(self):
        self.soon = None
        if self.lm is None:
            return
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if w < 50 or h < 50:
            return
        from PIL import ImageTk
        image, self.geometry = draw_entry(self.index, self.lm, (w, h), self.view)
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.itemconfigure(self.image_item, image=self.photo)
        self.canvas.tag_raise(self.message_item)
        self.drawn += 1

    # moving about

    def whole(self):
        return full_view(self.index, self.lm)

    def clamp(self, t0, t1):
        lo, hi = self.whole()
        width = t1 - t0
        if width >= hi - lo:
            return (lo, hi)
        if t0 < lo:
            t0, t1 = lo, lo + width
        if t1 > hi:
            t0, t1 = hi - width, hi
        return (t0, t1)

    def fit(self):
        if self.lm is not None:
            self.view = self.whole()
            self.redraw()

    def zoom(self, factor, frac=0.5):
        if self.lm is None or self.view is None:
            return
        t0, t1 = self.view
        at = t0 + (t1 - t0) * frac
        lo, hi = self.whole()
        width = min(max((t1 - t0) * factor, 0.25), hi - lo)
        self.view = self.clamp(at - width * frac, at - width * frac + width)
        self.redraw()

    def wheel(self, event):
        g = self.geometry
        if not g:
            return
        x = event.x_root - self.canvas.winfo_rootx()
        frac = min(max((x - g["x0"]) / float(g["x1"] - g["x0"]), 0.0), 1.0)
        self.zoom(0.8 ** (event.delta / 120.0), frac)

    def press(self, event):
        self.drag = (event.x, self.view)

    def move(self, event):
        g = self.geometry
        if not self.drag or not g or self.drag[1] is None:
            return
        x, (t0, t1) = self.drag
        shift = (event.x - x) / float(g["x1"] - g["x0"]) * (t1 - t0)
        self.view = self.clamp(t0 - shift, t1 - shift)
        self.redraw()

    def point(self, event):
        import numpy as np
        g = self.geometry
        if not g or self.lm is None:
            return
        x, y = event.x, event.y
        if not (g["x0"] <= x <= g["x1"] and g["py0"] <= y <= g["sy1"]):
            self.status.configure(text=" ")
            return
        lm = self.lm
        t = g["t0"] + (x - g["x0"]) / float(g["x1"] - g["x0"]) * (g["t1"] - g["t0"])
        if y > g["py1"]:
            near = int(np.count_nonzero(np.abs(lm["t"] - t) <= 0.5))
            self.status.configure(text="%s   ·   %s hashes in the second around here" % (clock(t, 1), "{:,}".format(near)))
            return
        f = (g["py1"] - y) / float(g["py1"] - g["py0"]) * g["fmax"]
        text = "%s   ·   %.2f kHz" % (clock(t, 2), f / 1000.0)
        if g["xs"].size:
            d = (g["xs"] - x) ** 2 + (g["ys"] - y) ** 2
            i = int(d.argmin())
            if d[i] <= 81:
                j = int(g["idx"][i])
                count = int(lm["count"][j])
                text += "      peak pair at %s: %.2f to %.2f kHz, %.0f ms apart   ·   hash %05X, in the table %s time%s%s" % (
                    clock(lm["t"][j], 2), lm["f1"][j] / 1000.0, lm["f2"][j] / 1000.0, lm["dt"][j] * 1000.0,
                    int(lm["hash"][j]), "{:,}".format(count), "" if count == 1 else "s",
                    " (its bucket keeps %d)" % self.index.depth if count > self.index.depth else "")
        self.status.configure(text=text)

    def open_source(self):
        if self.link:
            import webbrowser
            webbrowser.open(self.link["url"])

    # saving

    def save_picture(self):
        from tkinter import filedialog
        if self.lm is None:
            return
        k, name = self.entry
        target = filedialog.asksaveasfilename(parent=self.top, title="Save the picture", defaultextension=".png",
                                              initialfile="%s - hashes.png" % _safe_stem(name, "entry %d" % (k + 1)),
                                              filetypes=[("PNG picture", "*.png")])
        if not target:
            return
        header = ["%s   ·   %s" % (self.title_label.cget("text"), os.path.basename(self.index.path))]
        header += facts_text(self.index, self.facts).split("\n")
        image, _ = draw_entry(self.index, self.lm, (2400, 1250), self.view, header=header, scale=2.0)
        image.save(target)
        self.status.configure(text="Saved %s" % target)

    def save_csv(self):
        from tkinter import filedialog
        if self.lm is None:
            return
        k, name = self.entry
        target = filedialog.asksaveasfilename(parent=self.top, title="Save the hashes", defaultextension=".csv",
                                              initialfile="%s - hashes.csv" % _safe_stem(name, "entry %d" % (k + 1)),
                                              filetypes=[("CSV file", "*.csv")])
        if not target:
            return
        write_hashes_csv(self.lm, target)
        self.status.configure(text="Saved %s (%s hashes)" % (target, "{:,}".format(self.lm["t"].size)))

    def close(self):
        try:
            self.top.destroy()
        finally:
            self.on_close()


def register(remove=False):
    import winreg
    base = r"Software\Classes"
    script = os.path.abspath(__file__)
    pythonw = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
    if remove:
        for key in (PROG_ID + r"\shell\open\command", PROG_ID + r"\shell\open", PROG_ID + r"\shell", PROG_ID):
            try:
                winreg.DeleteKey(winreg.HKEY_CURRENT_USER, base + "\\" + key)
            except OSError:
                pass
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, base + r"\.pklz\OpenWithProgids", 0, winreg.KEY_SET_VALUE) as k:
                winreg.DeleteValue(k, PROG_ID)
        except OSError:
            pass
        print("removed from the Open with menu")
        return
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, base + "\\" + PROG_ID) as k:
        winreg.SetValueEx(k, None, 0, winreg.REG_SZ, "audfprint database")
        winreg.SetValueEx(k, "FriendlyTypeName", 0, winreg.REG_SZ, "audfprint database")
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, base + "\\" + PROG_ID + r"\shell\open") as k:
        winreg.SetValueEx(k, "FriendlyAppName", 0, winreg.REG_SZ, "PKLZ Viewer")
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, base + "\\" + PROG_ID + r"\shell\open\command") as k:
        winreg.SetValueEx(k, None, 0, winreg.REG_SZ, '"%s" "%s" "%%1"' % (pythonw, script))
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, base + r"\.pklz\OpenWithProgids") as k:
        winreg.SetValueEx(k, PROG_ID, 0, winreg.REG_NONE, b"")
    try:
        import ctypes
        ctypes.windll.shell32.SHChangeNotify(0x08000000, 0, None, None)     # SHCNE_ASSOCCHANGED
    except Exception:
        pass
    print('added to the Open with menu for .pklz: "%s" "%s" "%%1"' % (pythonw, script))


def run_gui(path, close_when_loaded=False, hash_test=None):
    import tkinter as tk
    from tkinter import filedialog, ttk

    root = tk.Tk()
    root.title("PKLZ Viewer")
    root.geometry("980x640")
    state = {"tracks": [], "links": [], "ok": None, "path": None}
    hashes = {"path": None, "index": None, "waiting": [], "building": False}   # this file's HashIndex
    shown = {"window": None}                                                  # the hashes window, when open

    top = ttk.Frame(root, padding=8)
    top.pack(fill="x")
    summary = tk.Text(top, height=9, wrap="word", relief="flat", background=root.cget("background"))
    summary.pack(fill="x")
    summary.insert("1.0", "No file opened.")
    summary.configure(state="disabled")

    bar = ttk.Frame(root, padding=(8, 0))
    bar.pack(fill="x")
    ttk.Label(bar, text="Filter:").pack(side="left")
    needle = tk.StringVar()
    ttk.Entry(bar, textvariable=needle, width=40).pack(side="left", padx=6)
    count_label = ttk.Label(bar, text="")
    count_label.pack(side="left")
    ttk.Button(bar, text="Open...", command=lambda: choose()).pack(side="right")
    ttk.Button(bar, text="Copy track list", command=lambda: copy()).pack(side="right", padx=6)
    ttk.Button(bar, text="Show hashes", command=lambda: show_hashes()).pack(side="right")
    ttk.Button(bar, text="Open source", command=lambda: open_source()).pack(side="right", padx=6)

    body = ttk.Frame(root, padding=8)
    body.pack(fill="both", expand=True)
    tree = ttk.Treeview(body, columns=("n", "name", "source", "hashes"), show="headings")
    for col, text, width, anchor in (("n", "#", 60, "e"), ("name", "Track", 620, "w"), ("source", "Source", 150, "w"), ("hashes", "Hashes", 100, "e")):
        tree.heading(col, text=text, command=lambda c=col: sort(c))
        tree.column(col, width=width, anchor=anchor, stretch=(col == "name"))
    scroll = ttk.Scrollbar(body, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scroll.set)
    tree.pack(side="left", fill="both", expand=True)
    scroll.pack(side="right", fill="y")
    order = {"col": "n", "down": False}

    def set_summary(text):
        summary.configure(state="normal")
        summary.delete("1.0", "end")
        summary.insert("1.0", text)
        summary.configure(state="disabled")

    def link_of(k):
        links = state["links"]
        return links[k] if k is not None and 0 <= k < len(links) else None

    def show_count():
        count_label.configure(text="%d of %d tracks" % (len(tree.get_children()), len(state["tracks"])))

    def fill(*_):
        text = needle.get().lower()
        rows = [(n, name, made, (link_of(n - 1) or {}).get("label", "")) for n, name, made in state["tracks"] if text in name.lower()]
        index = {"n": 0, "name": 1, "hashes": 2, "source": 3}[order["col"]]
        rows.sort(key=lambda t: t[index].lower() if index in (1, 3) else t[index], reverse=order["down"])
        tree.delete(*tree.get_children())
        for n, name, hashes_made, label in rows:
            tree.insert("", "end", values=(n, name or "(removed)", label, "{:,}".format(hashes_made)))
        show_count()

    def sort(col):
        order["down"] = (not order["down"]) if order["col"] == col else (col == "hashes")
        order["col"] = col
        fill()

    def copy():
        root.clipboard_clear()
        root.clipboard_append("\n".join(tree.item(i, "values")[1] for i in tree.get_children()))

    def copy_text(text):
        root.clipboard_clear()
        root.clipboard_append(text)

    def open_source(*_):
        k = selected_entry()
        link = link_of(k)
        if link is None:
            count_label.configure(text="Select an entry first" if k is None else "No source is known for this entry")
            root.after(3000, show_count)
            return
        import webbrowser
        webbrowser.open(link["url"])

    menu = tk.Menu(root, tearoff=0)

    def popup(event):
        row = tree.identify_row(event.y)
        if not row:
            return
        tree.selection_set(row)
        tree.focus(row)
        k = selected_entry()
        if k is None:
            return
        link = link_of(k)
        menu.delete(0, "end")
        if link:
            menu.add_command(label="Open source (%s)" % link["label"], command=open_source)
            menu.add_command(label="Copy source link", command=lambda: copy_text(link["url"]))
        else:
            menu.add_command(label="No source known", state="disabled")
        menu.add_separator()
        menu.add_command(label="Show hashes", command=show_hashes)
        menu.add_command(label="Copy name", command=lambda: copy_text(state["tracks"][k][1]))
        menu.tk_popup(event.x_root, event.y_root)

    tree.bind("<Button-3>", popup)

    # The hashes window: built on demand, follows the selection.

    def with_hashes(callback):
        """ callback(index, error) once the hashes of the open file are read (read once per file). """
        target = state["path"]
        if hashes["path"] == target and hashes["index"] is not None:
            callback(hashes["index"], None)
            return
        if hashes["path"] != target:
            hashes.update(path=target, index=None, waiting=[], building=False)
        hashes["waiting"].append(callback)
        if hashes["building"]:
            return
        hashes["building"] = True

        def work():
            try:
                result, error = HashIndex(target), None
            except Exception as problem:
                result, error = None, "%s: %s" % (type(problem).__name__, problem)
            root.after(0, hashes_read, target, result, error)
        threading.Thread(target=work, daemon=True).start()

    def hashes_read(target, result, error):
        if hashes["path"] != target:
            return      # another file was opened meanwhile
        hashes.update(index=result, building=False)
        waiting, hashes["waiting"] = hashes["waiting"], []
        for callback in waiting:
            callback(result, error)

    def selected_entry():
        sel = tree.selection()
        if not sel:
            return None
        n = int(str(tree.item(sel[0], "values")[0]).replace(",", ""))
        return n - 1 if 0 < n <= len(state["tracks"]) else None

    def step(delta):
        items = tree.get_children()
        if not items:
            return
        sel = tree.selection()
        i = items.index(sel[0]) + delta if sel else 0
        i = min(max(i, 0), len(items) - 1)
        tree.selection_set(items[i])
        tree.focus(items[i])
        tree.see(items[i])

    def show_hashes(*_):
        if not state["ok"]:
            return
        k = selected_entry()
        if k is None:
            items = tree.get_children()
            if not items:
                return
            tree.selection_set(items[0])
            tree.focus(items[0])
            k = selected_entry()
        if shown["window"] is None:
            shown["window"] = HashWindow(root, with_hashes, step, lambda: shown.update(window=None))
        shown["window"].show(k, state["tracks"][k][1], link_of(k))
        shown["window"].top.lift()

    def double_click(event):
        if tree.identify_region(event.x, event.y) == "cell":
            show_hashes()

    def selection_changed(_):
        if shown["window"] is not None:
            k = selected_entry()
            if k is not None:
                shown["window"].show(k, state["tracks"][k][1], link_of(k))

    tree.bind("<Double-Button-1>", double_click)
    tree.bind("<Return>", show_hashes)
    tree.bind("<<TreeviewSelect>>", selection_changed)

    def loaded(result, error):
        if error is not None:
            state["ok"] = False
            set_summary("Could not open this file:\n%s" % error)
        else:
            info, tracks, links = result
            state["ok"] = True
            state["tracks"] = tracks
            state["links"] = links
            set_summary("\n".join("%s: %s" % kv for kv in info.items()))
            fill()
            if hash_test is not None:
                root.after(200, hash_selftest)
        if close_when_loaded:
            root.after(300, root.destroy)

    def load(target):
        root.title("PKLZ Viewer - " + os.path.basename(target))
        set_summary("Loading %s ..." % target)
        if shown["window"] is not None:
            shown["window"].close()
        state["tracks"] = []
        state["links"] = []
        state["ok"] = None
        state["path"] = target
        fill()

        def work():
            try:
                info, tracks = read_pklz(target)
                links = [source_link(target, name) for _, name, _ in tracks]
                result, error = (info, tracks, links), None
            except Exception as problem:
                result, error = None, "%s: %s" % (type(problem).__name__, problem)
            root.after(0, loaded, result, error)
        threading.Thread(target=work, daemon=True).start()

    def choose():
        target = filedialog.askopenfilename(title="Open a PKLZ file", filetypes=[("audfprint database", "*.pklz"), ("All files", "*.*")])
        if target:
            load(target)

    # --selftest-hashes: open the biggest entry's hashes, zoom, move on, close.

    def hash_selftest():
        items = tree.get_children()
        biggest = max(range(len(state["tracks"])), key=lambda i: state["tracks"][i][2])
        tree.selection_set(items[biggest])
        tree.see(items[biggest])
        show_hashes()
        hash_test["started"] = True
        root.after(200, hash_selftest_wait, biggest, 0)

    def hash_selftest_wait(k, tries):
        window = shown["window"]
        if window is None or window.lm is None or not window.drawn:
            if tries > 600:
                hash_test["result"] = "the hashes were not drawn within 2 minutes"
                root.destroy()
                return
            root.after(200, hash_selftest_wait, k, tries + 1)
            return
        facts = window.facts
        whole = window.view
        window.top.attributes("-topmost", True)
        window.top.update()
        # The real events: a wheel notch over the middle, a drag to the left, the pointer on a peak.
        c = window.canvas
        mid_x, mid_y = c.winfo_width() // 2, c.winfo_height() // 3
        c.event_generate("<MouseWheel>", delta=480, x=mid_x, y=mid_y, rootx=c.winfo_rootx() + mid_x, rooty=c.winfo_rooty() + mid_y)
        wheeled = window.view
        c.event_generate("<ButtonPress-1>", x=mid_x, y=mid_y)
        c.event_generate("<B1-Motion>", x=mid_x - 200, y=mid_y)
        c.event_generate("<ButtonRelease-1>", x=mid_x - 200, y=mid_y)
        dragged = window.view
        g = window.geometry
        inside = [i for i in range(g["xs"].size) if g["x0"] + 2 <= g["xs"][i] <= g["x1"] - 2] if g else []
        if inside:
            c.event_generate("<Motion>", x=int(round(g["xs"][inside[0]])), y=int(round(g["ys"][inside[0]])))
        hash_test["pointer"] = window.status.cget("text")
        window.zoom(0.25)
        zoomed = window.view
        hash_test["moves"] = "wheel %.1f-%.1f s, drag %.1f-%.1f s" % (wheeled[0], wheeled[1], dragged[0], dragged[1])
        window.top.update()

        def shoot():
            try:
                grab()
            except OSError as problem:
                hash_test["shot_error"] = "no screenshot (%s: the screen may be locked)" % problem
            drawn_before = window.drawn
            step(1)
            root.after(400, finish, drawn_before)

        def grab():
            if hash_test.get("shot"):
                from PIL import ImageGrab
                w = window.top
                box = (w.winfo_rootx(), w.winfo_rooty(), w.winfo_rootx() + w.winfo_width(), w.winfo_rooty() + w.winfo_height())
                ImageGrab.grab(bbox=box, all_screens=True).save(hash_test["shot"])
                w.attributes("-topmost", False)
                w.lower()
                root.lift()
                root.update()
                box = (root.winfo_rootx(), root.winfo_rooty(), root.winfo_rootx() + root.winfo_width(), root.winfo_rooty() + root.winfo_height())
                ImageGrab.grab(bbox=box, all_screens=True).save(os.path.splitext(hash_test["shot"])[0] + "-main.png")

        def finish(drawn_before):
            # "Open source", from the main window's button and from the hashes window's link, with the
            # browser left out: only the addresses that would have been opened are noted.
            import webbrowser
            opened = []
            real_open = webbrowser.open
            webbrowser.open = lambda url, *a, **kw: opened.append(url) or True
            try:
                open_source()
                window.open_source()
            finally:
                webbrowser.open = real_open
            hash_test["source"] = "%s | would open: %s" % (window.source_label.cget("text") or "(no source line)", ", ".join(opened) or "nothing")
            moved = window.entry[0] if window.entry else None
            hash_test["result"] = "entry #%d: %d of %d hashes kept, whole view %.1f-%.1f s, zoomed to %.1f-%.1f s, " \
                                  "Down moved to entry #%s (%s, %d more drawing(s)), picture %dx%d" % (
                                      k + 1, facts["kept"], facts["made"], whole[0], whole[1], zoomed[0], zoomed[1],
                                      moved + 1 if moved is not None else "?", "kept %d" % window.facts["kept"] if window.facts else "?",
                                      window.drawn - drawn_before, window.canvas.winfo_width(), window.canvas.winfo_height())
            root.destroy()
        root.after(700, shoot)

    needle.trace_add("write", fill)
    if path:
        load(path)
    elif not close_when_loaded:
        root.after(100, choose)
    root.mainloop()
    return state


def main(argv):
    try:
        # Entry names hold characters a Windows console cannot show (yt-dlp's lookalikes).
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    if argv[:1] == ["--sources"]:
        for path in argv[1:]:
            _, tracks = read_pklz(path)
            found = 0
            for n, name, _ in tracks:
                link = source_link(path, name)
                if link:
                    found += 1
                    print("%d\t%s\t%s\t%s" % (n, link["label"], link["url"], name))
            print("# %s: %d of %d entries have a source link" % (os.path.basename(path), found, len(tracks)))
        return
    if argv[:1] == ["--register"]:
        return register()
    if argv[:1] == ["--unregister"]:
        return register(remove=True)
    if argv[:1] == ["--check"]:
        bad = 0
        for path in argv[1:]:
            try:
                info, tracks = read_pklz(path)
                print("OK   %s | %s tracks, %s hashes, %s" % (os.path.basename(path), info["Tracks"], info["Hashes stored"], info["Size on disk"]))
            except Exception as problem:
                bad += 1
                print("FAIL %s | %s: %s" % (os.path.basename(path), type(problem).__name__, problem))
        print("%d files, %d could not be opened" % (len(argv) - 1, bad))
        return
    if argv[:1] == ["--hashes"]:
        if len(argv) < 3:
            print("usage: pklz_viewer.pyw --hashes <file.pklz> <entry #|biggest> [picture.png]")
            return 2
        index = HashIndex(argv[1])
        k = index.biggest() if argv[2] == "biggest" else int(argv[2]) - 1
        if not 0 <= k < len(index.names):
            print("There is no entry #%s: the table has %d." % (argv[2], len(index.names)))
            return 2
        times, entry_hashes = index.entry(k)
        lm = landmarks(index, times, entry_hashes)
        facts = entry_facts(index, k, times, entry_hashes)
        title = "%s   ·   entry #%s" % (index.names[k] or "(removed entry)", "{:,}".format(k + 1))
        print(title)
        link = source_link(argv[1], index.names[k])
        if link:
            print("source: %s%s %s" % (link["label"], TIER_NOTE.get(link["tier"], ""), link["url"]))
        print(facts_text(index, facts))
        if len(argv) > 3:
            header = ["%s   ·   %s" % (title, os.path.basename(argv[1]))] + facts_text(index, facts).split("\n")
            image, _ = draw_entry(index, lm, (2400, 1250), None, header=header, scale=2.0)
            image.save(argv[3])
            print("picture: %s" % argv[3])
        return
    if argv[:1] == ["--selftest"]:
        state = run_gui(argv[1], close_when_loaded=True)
        print("window test: loaded=%s, %d tracks listed" % (state["ok"], len(state["tracks"])))
        return
    if argv[:1] == ["--selftest-hashes"]:
        test = {"shot": argv[2] if len(argv) > 2 else None}
        run_gui(argv[1], hash_test=test)
        print("hash window test: %s" % test.get("result", "did not finish"))
        print("  moves: %s" % test.get("moves", "-"))
        print("  pointer read-out: %s" % test.get("pointer", "-"))
        print("  source: %s" % test.get("source", "-"))
        if test.get("shot_error"):
            print("  %s" % test["shot_error"])
        return
    run_gui(argv[0] if argv else None)


if __name__ == "__main__":
    main(sys.argv[1:])

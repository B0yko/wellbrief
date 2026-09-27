# Air-gapped install

wellbrief never makes an outbound network call on its own (see the network guard in
`selfcheck`'s output below), so it installs and runs the same way with no internet access at
all. Two paths are verified below, both run for real (output trimmed, otherwise as printed):

- **Wheelhouse** — a folder of `.whl` files carried over on removable media.
- **Saved image** — a `docker save`/`docker load` tarball.

## Path 1: wheelhouse

On a connected machine, build the wheel and download its one dependency as a wheel too:

```
$ uv build --out-dir dist
Building source distribution...
Building wheel from source distribution...
Successfully built dist/wellbrief-0.1.0.tar.gz
Successfully built dist/wellbrief-0.1.0-py3-none-any.whl

$ mkdir wheelhouse
$ cp dist/wellbrief-0.1.0-py3-none-any.whl wheelhouse/
$ pip download 'pypdf>=6.19.0' -d wheelhouse --no-deps
Collecting pypdf>=6.19.0
  Using cached pypdf-6.19.0-py3-none-any.whl (395 kB)
Saved ./wheelhouse/pypdf-6.19.0-py3-none-any.whl
Successfully downloaded pypdf

$ ls wheelhouse
pypdf-6.19.0-py3-none-any.whl
wellbrief-0.1.0-py3-none-any.whl
```

Copy `./wheelhouse` (two files) to the air-gapped machine by USB drive, internal file share, or
any offline transfer, keeping the same folder name. Install with no index at all:

```
$ pip install --no-index --find-links ./wheelhouse wellbrief
```

Verified end to end in a container with no network access, to prove the two wheels are
sufficient and nothing else gets pulled in:

```
$ docker run --rm --network none -v "./wheelhouse:/wheelhouse:ro" python:3.12-slim sh -c \
    "pip install --no-index --find-links /wheelhouse wellbrief && wellbrief selfcheck"
Looking in links: /wheelhouse
Processing /wheelhouse/wellbrief-0.1.0-py3-none-any.whl
Processing /wheelhouse/pypdf-6.19.0-py3-none-any.whl (from wellbrief)
Installing collected packages: pypdf, wellbrief
Successfully installed pypdf-6.19.0 wellbrief-0.1.0
wellbrief selfcheck (seed 20260731)
  corpus generate    0.144s
  ingest             1.467s
  index              1.127s
  eval              12.915s
  brief verify       0.062s
outbound connection attempts: 0
guard self-test: blocked 1/1
selfcheck: PASS
```

Exit code `0`, `--network none` the whole time: the install and the `selfcheck` run (which
itself generates a corpus, ingests it, builds indexes, and runs the offline eval suite) never
touch the network.

## Path 2: saved image

Anywhere the image already exists (built locally, or pulled from a registry while still
connected), save it to a single tar file:

```
$ docker build -t wellbrief:local .
$ docker save wellbrief:local -o wellbrief-local.tar
```

Copy `wellbrief-local.tar` (tens of megabytes) to the air-gapped machine the same way, then
load and run it there with no registry involved:

```
$ docker load -i wellbrief-local.tar
Loaded image: wellbrief:local

$ docker run --rm --network none wellbrief:local selfcheck
wellbrief selfcheck (seed 20260731)
  corpus generate    0.146s
  ingest             1.611s
  index              1.137s
  eval              12.346s
  brief verify       0.057s
outbound connection attempts: 0
guard self-test: blocked 1/1
selfcheck: PASS
```

This was run after `docker rmi wellbrief:local` on the same machine (so `docker load` was
loading from the tar, not reusing a cached image) and still passes with `--network none`. A
released build works the same way once pulled from its registry once while connected:
`docker save ghcr.io/b0yko/wellbrief:0.1.0 -o wellbrief-0.1.0.tar`, then `docker load` and
`docker run --network none ... selfcheck` on the air-gapped side.

## Pointing it at a site's own files

Neither path above touches a site's actual well files; those stay on the air-gapped machine
and go in through a read-only bind mount plus `ingest`:

```
$ docker run --rm --network none \
    -v "/path/to/site-archive:/data:ro" \
    -v "wellbrief-home:/opt/wellbrief-home" \
    wellbrief:local ingest /data --field "Site A"
```

`ingest` reads every `.txt`/`.md`/`.pdf`/`.docx`/`.csv` file under the mounted folder and
writes the resulting store and indexes into the workspace, never back into `/data` (mounted
`:ro` above, which is not required but documents the intent). Running the same `ingest`
again over an updated archive only reparses files whose content changed.

### Where the workspace lives

The workspace is everything under `$WELLBRIEF_HOME/<name>/`: the SQLite store, the per-field
BM25 and vector index files, and the audit/egress logs. `$WELLBRIEF_HOME` defaults to
`~/.wellbrief` and is overridden by the `WELLBRIEF_HOME` environment variable (the image sets
it to `/opt/wellbrief-home`); the workspace name defaults to `default` and is overridden by
`--workspace` or `WELLBRIEF_WORKSPACE`. Mount a volume there (as `wellbrief-home` above) to
keep the ingested data across container restarts.

### Mapping a site's own report template

Sites that use a different daily-report layout, different end-of-well-report section
headings, or their own short NPT codes do not need any code changes: a `wellbrief.toml` next
to the files (or passed with `ingest --config`) remaps the labels, section headings, and
taxonomy aliases `ingest` looks for. See `examples/alt-template/wellbrief.toml` for a worked
example and the settings reference for every key it understands.

### Serving the UI

`wellbrief serve` (and `wellbrief demo`) bind to `127.0.0.1` by default, so the JSON API and
web UI are reachable only from the same machine unless `--host` is set explicitly, at which
point the CLI prints a warning that the API has no authentication of its own.

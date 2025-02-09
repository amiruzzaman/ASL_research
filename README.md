# ASL Research

Repository for ASL 2 English and English 2 ASL research.

## Development

We use [just](https://github.com/casey/just#installation) to run commands.
It isn't required but will contain shortcuts for the commands outlined in this readme.

### Installing NPM Deps

NPM Deps are still managed imperatively, so you need to install them manually.

To do so use the `npm` just recipe

```sh
just npm i
```

...or cd manually and run install. You _will_ need `npm` installed on your
host system for this to work.

```sh
cd web-frontend
npm i
cd ..
```

### With Nix/Lix

The easiest way to get set up is to [install Lix](https://lix.systems/install/).

Lix is a declarative, functional package manager and build tool geared towards
reporoducible builds.

On Windows, I'd recommend getting it installed in a WSL container.

After installing both Lix and just, you can start a development environment by
simply running the `dev` recipe.

```sh
just dev
```

If you don't have `just`, you can run the command directly.

```sh
nix develop --command "mprocs --names 'Backend,Frontend' 'python src/web_backend/__init__.py' 'cd web-frontend; npm run dev'"
```

### Imperative

Another way to get set up is to imperatively install all dependencies.

You will need the following system dependencies:

- [FFMPEG](https://ffmpeg.org/)
- [OpenCV](https://opencv.org/)
- [UV](https://docs.astral.sh/uv/getting-started/installation/)
- [NodeJS](https://nodejs.org/en)

We use the `uv` tool to manage python dependencies, to create a virtualenv with
everything simply run `uv sync`.

We use `npm` for Node dependencies, `cd` into `web-frontend` and run
`npm install` to get everything installed.

Now, you will need to run both the backend and frontend separately. First run
the backend.

```sh
uv run dev
```

And on another terminal start the frontend.

```sh
npm run dev
```

### Accessing The Dev Server

The **frontend** terminal should give a link to `http://localhost:4321`, if you
ran the `just dev` recipe use the arrow keys to select the frontend
output within `mprocs`.

This is where you'll access the frontend site, it will communicate with the backend
(running on port `5000`) via AJAX. In production these two webservers are
expected to be unified by a reverse proxy.

Both the frontend and backend are set up to reload on file change, double check
any errors that may be appearing in the `mprocs` UI.

### Managing Dependencies

For python, run the `uv` recipe justfile recipe to add a package.

```sh
just uv add NAME
```

Or run it without Nix/Just.

```sh
uv add NAME
```

---

For nodejs, do the same with `npm`.

```sh
just npm add NAME
```

Or cd and run without Nix/Just.

```sh
cd web-frontend
npm add NAME
```

For both `npm` and `uv` change `add` to `remove` etc. as needed.
Run `uv --help` or `npm --help` for more info.

### Additional Just Recipes

For Nix specifically, you can run `just shell` to get into a dev shell with
all dependencies loaded (`nix develop`).

You can also run `just run COMMAND` to run a command within the dev shell
(`nix develop --command`).

Finally, run `just` without any arguments for a list of recipes available.

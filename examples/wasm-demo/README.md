# Demo of `cairo-rs` on WebAssembly

While cairo-rs is compatible with WebAssembly, it doesn't implement any bindings to it.
Instead, create a new WebAssembly crate with cairo-rs as a dependency and implement the required functionality there.

Since mimalloc is not automatically compilable to WebAssembly, the cairo-rs dependency should disable the default features, which will in turn disable mimalloc.

A working example is provided in this repository.

## Dependencies

To compile and run the example you need:

- a Cairo 0 compiler
- the _wasm-pack_ crate
- some HTTP server (for example: the `live-server` npm module)

> **Note**
> The first two dependencies can be installed via the repository's installation script (see ["Installation script"](../../README.md#installation-script))

## Building

To build the example, first compile your Cairo program:

```sh
cairo-compile src/array_sum.cairo --no_debug_info --output src/array_sum.json
```

And then the WebAssembly package:

```sh
wasm-pack build --target=web
```

This will generate a javascript module that is directly loadable by the browser.

## Running

To run the example webpage, you need to run an HTTP server.
For example, using the _live-server_ npm module:

```sh
# while in <repo>/examples/wasm-demo
npx live-server
```

> **Warning**
> Trying to run `index.html` directly (i.e. URL starts with `file://`) will result in a CORS error.

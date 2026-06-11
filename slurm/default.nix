with import <nixpkgs> { config = { allowUnfree = true; }; };


mkShell {
  venvDir = "./_venv";
  # Add dependencies that pip can't fetch here (or that we don't want to
  # install using pip).
  buildInputs = [
    pkgs.python312
    pkgs.stdenv.cc.cc
  ] ++ (with pkgs.python312Packages; [
    venvShellHook
    wheel
    pip
  ])
     ++ (import ./system-dependencies.nix { inherit pkgs; });
  postShellHook = ''
    unset SOURCE_DATE_EPOCH
    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [pkgs.stdenv.cc.cc]}:${pkgs.lib.makeLibraryPath [pkgs.zlib]}:$LD_LIBRARY_PATH"
    export PYTHONPATH=$venvDir/${python312.sitePackages}:$PYTHONPATH
  '';
  postVenvCreation = ''
    unset SOURCE_DATE_EPOCH
    pip install --upgrade pip
    pip install -r requirements.txt
  '';
}
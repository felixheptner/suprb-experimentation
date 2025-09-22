with import (builtins.fetchGit {
  url = "https://github.com/NixOS/nixpkgs/";
  rev = "35ad3c79b6c264aa73bd8e7ca1dd0ffb67bd73b1";
}) { config = { allowUnfree = true; }; };

mkShell {
  venvDir = "./_venv";
  venvPython = pkgs.python39.withPackages (ps: [ ps.pip ps.setuptools ps.wheel ]);

  buildInputs =
    (with pkgs.python39Packages; [ venvShellHook ])
    ++ [
      pkgs.gcc
      pkgs.stdenv.cc.cc
      pkgs.zlib
    ]
    ++ (import ./system-dependencies.nix { inherit pkgs; });

  postShellHook = ''
    unset SOURCE_DATE_EPOCH
    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [pkgs.stdenv.cc.cc pkgs.zlib pkgs.gcc]}:$LD_LIBRARY_PATH"
    export PYTHONPATH=$venvDir/${pkgs.python39.sitePackages}:$PYTHONPATH
  '';

  postVenvCreation = ''
    unset SOURCE_DATE_EPOCH
    # Upgrade pip inside the venv
    $venvDir/bin/python -m pip install --upgrade pip setuptools wheel
    # Install project requirements (overwrite existing dirs automatically)
    PIP_EXISTS_ACTION=w $venvDir/bin/pip install -r requirements.txt
  '';
}

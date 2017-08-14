# proton
Proton practical programming language

To run, you will need Python 3. All modules used should come with the standard Python distribution (`operator`, `sys`, `builtins`, `math`, `ast`, and `traceback`).

To run a Proton program, just run `./proton <filename> [args...]`. If you are on Windows, you may need to change it to `proton.py` and run it with `python proton.py <filename> [args...]`.

    usage: proton [-c <cmd> | -r <recursion_limit> | -h] [file] [args...]
    Options:
    -c <cmd>: Run <cmd> as Proton code and do not enter the shell
    -r <recursion_limit>: Set the recursion limit to <recursion_limit>
    -h: Display this help message
    If no file name or <cmd> is given, this will enter a Proton shell

# -*- coding: utf-8 -*-
"""Pull needed files from the AutoDL instance (py3.6-safe)."""
import os, sys, warnings
warnings.filterwarnings("ignore")
import paramiko

HOST, PORT, USER = "connect.westb.seetacloud.com", 32710, "root"
KEY = os.path.expandvars(r"%USERPROFILE%\.ssh\autodl_5090")
PWD = os.environ.get("AUTODL_SSH_PWD", "qduT2K1hqtTI")

def client():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    if PWD:
        c.connect(HOST, port=PORT, username=USER, password=PWD, timeout=20,
                  look_for_keys=False)
    else:
        c.connect(HOST, port=PORT, username=USER, key_filename=KEY, timeout=20)
    return c

def main():
    cmd = sys.argv[1]
    c = client()
    if cmd == "run":
        stdin, stdout, stderr = c.exec_command(sys.argv[2],
                                               timeout=int(sys.argv[3]) if len(sys.argv) > 3 else 300)
        out = stdout.read().decode("utf-8", "replace")
        err = stderr.read().decode("utf-8", "replace")
        code = stdout.channel.recv_exit_status()
        sys.stdout.write(out)
        if err.strip():
            sys.stdout.write("\n[stderr] " + err[-1500:] + "\n")
        sys.stdout.write("\n[exit] %d\n" % code)
    elif cmd == "get":
        s = c.open_sftp()
        s.get(sys.argv[2], sys.argv[3])
        s.close()
        sys.stdout.write("[downloaded] %s\n" % sys.argv[3])
    c.close()

if __name__ == "__main__":
    main()

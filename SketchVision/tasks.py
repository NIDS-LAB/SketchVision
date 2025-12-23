from invoke import task 

@task
def build(c):
   c.run("./script/rebuild.sh")

@task
def compile(c):
    c.run("./script/complie.sh")

@task
def run_test(c):
    c.run("cd build && ../script/run_test.sh && cd ..")

@task
def run_bot(c):
    c.run("cd build && ../script/run_slow_bot.sh && cd ..")

@task
def run_c2(c):
    c.run("cd build && ../script/run_slow_c2.sh && cd ..")

@task
def run_ddos(c):
    c.run("cd build && ../script/run_slow_ddos.sh && cd ..")

@task
def run_exfil(c):
    c.run("cd build && ../script/run_slow_exfil.sh && cd ..")

@task
def run_surveil(c):
    c.run("cd build && ../script/run_slow_surveil.sh && cd ..")

@task
def run_all(c):
    c.run("cd build && ../script/run_slow_bot.sh && cd ..")
    c.run("cd build && ../script/run_slow_c2.sh && cd ..")
    c.run("cd build && ../script/run_slow_ddos.sh && cd ..")
    c.run("cd build && ../script/run_slow_exfil.sh && cd ..")
    c.run("cd build && ../script/run_slow_surveil.sh && cd ..")

@task
def clean(c):
    c.run("./script/clean.sh")

#!/usr/bin/expect -f

proc usage {} {
    puts stderr "Usage: ssh_secret_expect.tcl --host <host> --user <user> --port <port> -- <command...>"
    exit 2
}

set host ""
set user ""
set port "22"
set command {}

set argcnt [llength $argv]
set i 0
while {$i < $argcnt} {
    set arg [lindex $argv $i]
    if {$arg eq "--"} {
        incr i
        set command [lrange $argv $i end]
        break
    }
    if {$arg eq "--host"} {
        incr i
        if {$i >= $argcnt} { usage }
        set host [lindex $argv $i]
    } elseif {$arg eq "--user"} {
        incr i
        if {$i >= $argcnt} { usage }
        set user [lindex $argv $i]
    } elseif {$arg eq "--port"} {
        incr i
        if {$i >= $argcnt} { usage }
        set port [lindex $argv $i]
    } else {
        usage
    }
    incr i
}

if {$host eq "" || $user eq "" || [llength $command] == 0} {
    usage
}

set secret_tool [auto_execok secret-tool]
if {$secret_tool eq ""} {
    puts stderr "secret-tool not found in PATH."
    exit 3
}

if {[catch {
    set password [exec $secret_tool lookup service lerobot-gui protocol ssh host $host user $user port $port]
} err]} {
    puts stderr "Unable to load SSH password from secret store: $err"
    exit 4
}

set password [string trim $password]
if {$password eq ""} {
    puts stderr "No SSH password found in secret store for $user@$host:$port."
    exit 5
}

set timeout -1
set prompt_hits 0

eval spawn -noecho -- $command

expect {
    -re "(?i)are you sure you want to continue connecting.*" {
        send -- "yes\r"
        exp_continue
    }
    -re "(?i)(password|passphrase).*: *$" {
        if {$prompt_hits > 3} {
            puts stderr "Too many password prompts."
            exit 6
        }
        send -- "$password\r"
        incr prompt_hits
        exp_continue
    }
    timeout {
        puts stderr "SSH command timed out waiting for prompt/output."
        exit 124
    }
    eof {
        catch wait result
        set code [lindex $result 3]
        if {$code eq ""} {
            set code 0
        }
        exit $code
    }
}

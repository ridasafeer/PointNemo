# Point Nemo

## Dependencies

- make
- cmake
- gnu compiler

---
# Setting Up SSH Keys for GitHub (+ RBP too possibly)


Steps to set up your SSH Keys on your local to securely to connect to GHE, will also be useful when SSHing into Raspberry Pi

---

## Step 1: Check for existing SSH keys

Open a terminal (ideally Git Bash) and run:

```bash
ls -al ~/.ssh
```

If you see files like `id_ed25519` and `id_ed25519.pub` (or `id_rsa` / `id_rsa.pub`), you already have SSH keys.  
If not, proceed to the next step.

---

## Step 2: Generate a new SSH key

Run the following command (replace w your github associated email address):

```bash
ssh-keygen -t ed25519 -C "xud48@mcmaster.ca"
```

When prompted:
- **File to save key:** Press **Enter** to accept the default store location (`~/.ssh/id_ed25519`)
- **Passphrase:** Optional, adds security (not needed)

This creates two files:
- `~/.ssh/id_ed25519` → Private key (keep this safe)
- `~/.ssh/id_ed25519.pub` → Public key (will be uploaded to GitHub)

---

## Step 3: Activate the SSH agent and add your key

Start the SSH agent in the background:
```bash
eval "$(ssh-agent -s)"
```

Then add your private key:
```bash
ssh-add ~/.ssh/id_ed25519
```

The **SSH agent** securely stores your key in memory, so Git can use it automatically without retyping your passphrase.

---

## Step 4: Add your public key to GitHub

CAT out your public key:
```bash
cat ~/.ssh/id_ed25519.pub
```

Copy the entire line that begins with `ssh-ed25519` or `ssh-rsa`.

Then:
1. Go to **GitHub → Settings → SSH and GPG keys → New SSH key**
2. Give it a title (e.g., *laptop ssh key*)
3. Paste the key
4. Click **Add SSH key**

---

## Step 5: Switch your repo to SSH

Check your current remote URL:
```bash
git remote -v
```

If it shows HTTPS, switch to SSH:
```bash
git remote set-url origin git@github.com:USERNAME/REPOSITORY.git
```

Replace `USERNAME` and `REPOSITORY` with your actual GitHub username and repository name.

---

## Step 6: Test connection

Run this your connection:
```bash
ssh -T git@github.com
```

You should see something like:
```
Hi user! You've successfully authenticated...
```
---

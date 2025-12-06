# Deployment Instructions

## Option 1: Force Push (Quick - Replaces Everything)

```bash
cd "/Users/julia/Downloads/bsc thesis /git_new version"
git init
git remote add origin https://github.com/vitjuli/electronic-nose-ml.git
git add .
git commit -m "Complete repository restructure"
git push -f origin main  # or 'master' if that's your default branch
```

## Option 2: Careful Approach (Step-by-Step)

### Step 1: Initialize and Connect
```bash
cd "/Users/julia/Downloads/bsc thesis /git_new version"
git init
git remote add origin https://github.com/vitjuli/electronic-nose-ml.git
```

### Step 2: Check Remote Connection
```bash
git remote -v
# Should show:
# origin  https://github.com/vitjuli/electronic-nose-ml.git (fetch)
# origin  https://github.com/vitjuli/electronic-nose-ml.git (push)
```

### Step 3: Add Files and Commit
```bash
git add .
git status  # Review what will be committed
git commit -m "Complete repository restructure

- Reorganized into two main projects: interpretability-analysis and calibration
- Created professional Python package structure with src/ modules
- Added comprehensive documentation (README, QUICKSTART, CONTRIBUTING)
- Implemented all feature importance methods
- Implemented calibration methods
- Added setup.py for package installation
- Included citation information and MIT license"
```

### Step 4: Push (Choose One)

**Option A: Force push to completely replace**
```bash
git push -f origin main
```

**Option B: Create a new branch first (safer)**
```bash
git checkout -b restructure
git push origin restructure
# Then create a Pull Request on GitHub and merge
```

## Troubleshooting

### Error: "remote origin already exists"
```bash
git remote remove origin
git remote add origin https://github.com/vitjuli/electronic-nose-ml.git
```

### Error: "authentication failed"
You need a Personal Access Token:
1. Go to: https://github.com/settings/tokens
2. Generate new token (classic)
3. Select scope: `repo` (full control of private repositories)
4. Copy the token
5. Use it as password when prompted

### Error: "src ref main does not match any"
Your default branch might be 'master':
```bash
git push -f origin master
```

### Check which branch to use
```bash
git ls-remote --heads origin
# This will show all branches on remote
```

## After Pushing

1. Visit: https://github.com/vitjuli/electronic-nose-ml
2. Verify all files are there
3. Check that README.md displays correctly
4. Test the installation instructions:
   ```bash
   pip install -e .
   ```

## Authentication Setup (One-time)

### Using Personal Access Token (Recommended)
```bash
# After entering username, use the token as password
# To cache credentials:
git config --global credential.helper cache
# Or store permanently (less secure):
git config --global credential.helper store
```

### Using SSH (More Secure)
```bash
# Change remote URL to SSH
git remote set-url origin git@github.com:vitjuli/electronic-nose-ml.git
```

## Files That Will Be Uploaded

- README.md (main project description)
- QUICKSTART.md (quick start guide)
- CONTRIBUTING.md (contribution guidelines)
- LICENSE (MIT license)
- CITATION.cff (citation metadata)
- setup.py (package installation)
- requirements.txt (dependencies)
- .gitignore (ignore patterns)
- interpretability-analysis/ (feature importance project)
- calibration/ (calibration project)
- data/ (data directory with README)
- docs/ (documentation)
- models/ (saved models directory)

## Contact

If you encounter issues:
- Check GitHub status: https://www.githubstatus.com/
- Review Git documentation: https://git-scm.com/doc
- Email: vityugova.julia@physics.msu.ru

# code-ocean-capsule-template

Personalized, basic structure of a Code Ocean capsule.

Based on [aind-capsule-template](https://github.com/AllenNeuralDynamics/aind-capsule-template)
with the following customizations:
- **/LICENSE:** Allen Institute custom license (AIND uses MIT license)
- **/environment/postInstall_template:** contains code to install an updated version of code-server (open source VS Code) + a set of useful VS Code extensions (e.g. Python, Jupyter, autoDocstring, Git Graph)
- **/code/vscode_settings_template.json:** contains a few key settings to configure your VS Code sessions (e.g. color theme, git askpass, ruler)

## Usage
Instructions for how to use this GitHub template to create a new Code Ocean capsule.

### A. Create a new GitHub repo from this template
1. On this template's homepage, https://github.com/meghanaturner/code-ocean-capsule-template/, click on the green `Use this template` button in the upper right and select `Create new repository` from the dropdown menu.
2. Give your repo a name & description. Decide whether the owner will be your personal account or the AllenInstitute organization.
    1. If you make the owner the AllenInstitute organization, then you also need to decide if it will be public, internal, or private. For most capsules, I default to private & share it directly with any collaborators I want to have access so I don't clog up the AllenInstitute repo list with a bunch of abandoned exploratory analysis capsules.
3. Edit metadata to reflect your new repo's details in the following files:
    1. `/metadata/metadata.yml` - edit name, description, authors (this will make sure your new CO capsule starts with the correct metadata info)
    2. `/README.md` - edit title & to remove these instructions

### B. Create a new Code Ocean capsule from the new repo
3. In Code Ocean (https://codeocean.allenneuraldynamics.org/), create a new capsule:
    1. `Create` (circled plus &#8853; symbol in lefthand menu)
    2. Use the `Capsule: Clone from Git` option
    3. copy+paste the URL to the new repo you created in (A) into the box (Note: you *might* need to add `.git` to the end of the URL in order for this to work) -> click `Confirm`
4. In order to create a CO capsule from a repo, you must have your GitHub credentials added to your CO account:
    1. Go to `Account` (person symbol at bottom of lefthand menu) -> `Credentials` -> &#8853; `Add Credentials` (blue button, upper right)
    2. Select `GitHub Credentials` from dropdown menu & follow instructions in the pop-up window
  
### C. Install a newer version of VS Code via postInstall script
5. In the `Environment` GUI of your capsule, scroll down to the `Post-Install Script` section and click the blue hyperlink `Edit Post-Install Script`
6. Copy the contents of the `Environment > postInstall_template` file into the `postInstall` file that was just created by the GUI
7. Start up a VS Code cloud workstation. postInstall will run as part of the environment build and should install the newer code-server version & extensions.

### D. Configure your VS Code preferences
Since the `postInstall` file includes instructions to store the code-server installation in our `/capsule/code/` directory, which is stored in your new GitHub repo, you should only have to do these steps once; your preferences will persist across any subsequent workstation sessions.

8. To quickly add a few key preferences in a single step, while in your VS Code cloud workstation:
    1. Copy the contents of `/code/vscode_settings_template.json` into `/capsule/code/.vscode/User/settings.json`
    2. You may need to reload the CO window for this change to take effect.

9. Alternatively, you can set any preferences you'd like manually. For example, three preferences that are set in `/code/vscode_settings_template.json` are described in detail below:
    1. Change Color Theme: `F1` -> start typing `Preferences: Color Theme` & select when it pops up via autocomplete -> e.g. select “Dark (Visual Studio)” for dark theme
    2. Add a 80 character Ruler to all text editor windows (including Jupyter notebooks):
        1. Open the Settings UI (`File` -> `Preferences` -> `Settings`, OR `Ctrl+,`)
        2. Search for “Editor: Rulers” > click on `Edit in settings.json`
        4. Add the item: `"editor.rulers": [80]`
    4. Stop VS Code from making you sign in to GitHub every. single. session.
        1. Open the Settings UI (`File` -> `Preferences` -> `Settings`, OR `Ctrl+,`)
        2. Search for “Git: Use Integrated Ask Pass"
        3. Uncheck the box for “Git: Use Integrated Ask Pass / Controls whether GIT_ASKPASS should be overwritten to tuse the integrated version"  (This setting is checked by default and is telling code-server to override the intended mount code-ocean behavior where your GitHub credentials are stored as a token credential.)
        4. You may need to reload the CO window for this change to take effect
    5. Install any additional Extensions not specified in the `postInstall` script
  
10. (Optional) Install GitHub Copilot
    1. See the relevant section in AIND's Code Ocean Best Practices doc (can be found in the Code Ocean Teams channel under the Files tab) for details

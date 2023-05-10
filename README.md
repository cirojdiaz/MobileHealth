# Mobile Health App
This is where we can upload our files that we are working on for the Mobile Health Project for Winter 2023. 


## General Overview

*Look into Onboarding documents inside the Mobile Health App Project folder on Google Drive.*

We’ve been developing an app that aims to facilitate the transfer of self-monitored data from the patient to the physician. 
Features/milestones from this app are:
- Selecting Flutter SDK as our app development software (Firebase for backend)
- Autonomous data entry using optical character recognition (OCR)
- Prediction using self reported data with neural networks (python for ML)

The project diverges into a few areas:
- Machine Learning
- Health Research
- App Development


**Files**
Each directory should have their own readme.md files. This is to prevent the main readme file from getting too big or too long.

**Machine Learning** <br/>

---
## Getting Started

### VS Code

Visual Studio (VS) Code is a comprehensive and effective integrated development environment (IDE) that our team will be using. While you are welcome to continue using a different IDE or the one you are already familiar with, it may be more beneficial for the team to use a single IDE so that we can troubleshoot and resolve any issues that may arise more efficiently, as some problems may be specific to a particular IDE. If you do not already have VS installed, please consider downloading it to ensure consistency across the team.

To download VS Code, you can go to this link to download. 
- (https://code.visualstudio.com/download)


### Github
This project is collaborated with other people on the team by using GitHub. <br />
If you already have a GitHub account then please sign in and make sure that you have access to the Mobile Health GitHub 
by asking your Research Lead and Research Coordinator.<br />

If you do not have GitHub account, here are the steps to create a GitHub account:<br />
If you are having trouble checkout GitHub's official site for guidance, [Set up Git](https://docs.github.com/en/get-started/quickstart/set-up-git)
1. Go to [Github](https://github.com/) and sign up.
2. Ask your research lead to add you as a collaborator if you do not yet have access to the project
5. Once you have access to the proper folders you need to download git. <br />
   Please checkout [Git Download](https://git-scm.com/downloads) if you are having troubles <br />
   ***Mac***<br />
      1. Install Homebrew if you don't have it already, open up terminal:
      ```commandline
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
      ```
      2. Using Homebrew install Git
      ```commandline
        brew install git
      ```
   ***Windows***<br />
      Here is a detailed description of [downloading Git for Windows](https://phoenixnap.com/kb/how-to-install-git-windows)
      1. Goto [Git Download](https://git-scm.com/downloads)
      2. Click Windows
      3. Download the .exe file
      4. Open up the .exe file and follow the instructions to download Git.
7. Set up git with your username and email.
   1. Open terminal/shell and type: <br />
   ```
        git config --global user.name "Your name here"
        git config --global user.email "your_email@example.com"
   ```
   2. (Optional) 
   ```commandline
        git config --global color.ui true
        git config --global core.editor emacs
   ```
   First one will enable colour output in terminal and second will tell git you want to use emacs.

### Downloading Python and Pip
**Python version 3.9**

For future people working on this project, I highly recommend that everyone uses the same python version.
As people on different Python versions, might cause different issues.

***Mac***
1. First download Homebrew (Check above for downloading Homebrew)
2. Using Homebrew download Python
```commandline
brew install python
```
3. Check if the Python is installed correctly, try the following command:
```commandline
python --version
```
If you are looking to download 3.9 you can head over to [Download Python Versions](https://www.python.org/downloads/macos/)
and look for Python 3.9.0 or latest 3.9 version. 

When you download Python using Homebrew it should automatically come with pip installed.<br />
If not then use the following command (You must have Python 3.4 or higher):
```commandline
python3 -m ensurepip
```

***Windows***<br />
[More Helpful Guide](https://phoenixnap.com/kb/how-to-install-python-3-windows)
1. Go to [download python](https://www.python.org/downloads/windows/) on the official website.
2. Download 3.9.0 (or anything higher in 3.9) .exe file.
3. Run the .exe file.
4. Follow the instructions

To download Pip <br />
[More Helpful Guide](https://phoenixnap.com/kb/install-pip-windows)
1. Launch command prompt.
2. Using command prompt, download get-pip.py file:
```commandline
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
```
3. To install pip type the following command:
```commandline
python get-pip.py
```
4. To verify that you have downloaded pip properly:
```commandline
pip help
```
5. Add pip to Windows Environment Variables <br />
To run pip you need to add it to Windows environment variables to avoid getting the "not on PATH" error.
   1. Open System and Security window by searching it for it in the Control Panel.
   2. Navigate to System settings.
   3. Select Advanced system settings.
   4. Open the Environment Variables and double-click on Path variable in the System Variables.
   5. Select New and add the directory where you installed pip.
   6. Click OK to save the changes.
   
### List of Imports needed for this project

*Machine Learning*
- scipy
- numpy
- statistics
- pandas
- cv2
- numpy.lib.stride_tricks
- matplotlib
- datarecovplots

As you get started onto the project there are number of files that you must read. These files are located inside
Mobile Health App Project directory inside Google Drive that the Research Coordinator have shared with you. If you do not have
access to these files please contact your Research Coordinator or Research Lead and get the proper access. Below is a link to a Mobile Health App 
presentation from a couple terms ago <br />

List of files that needs to be read for understanding the project

- [ML Documentation](https://docs.google.com/document/d/1Cfow0BIWsX0VMSSymRksHtagrC2eursOoVQTWkXraMY/edit#heading=h.v9m0zir4agaw)
- [ML Tutorials](https://docs.google.com/document/d/1fAgJmrp_d_AfqKqlq8Lm7MTvcaoX4i2Rqcu0Up8kFHg/edit)
- [Project Link](https://docs.google.com/presentation/d/119Y7xxxx2jpIRhqcYj-XFr1qBst-LvgzeqU0me6cDA4/edit#slide=id.p)

### Git Clone Project
1. Create a directory(Folder) inside your computer.
2. Open terminal(if Mac)/command prompt(if Windows).
3. Navigate to your directory. <br />
   ***Mac***<br />
   [Navigating Terminal](https://riptutorial.com/terminal/example/26023/basic-navigation-commands) <br />
   ***Windows***<br />
   [Navigating Command Prompt](https://www.digitalcitizen.life/command-prompt-how-use-basic-commands/)
4. Type the following command:
```commandline
git init
```
5. Type the following command to clone the Mobile_Health_Upload repository:
```commandline
git clone <link to the github project>
```
6. Follow the prompt on the terminal/command prompt.

#### If you are having any troubles downloading OR getting set up, please talk to your Research Lead for guidance.


---
## Note to Future Research Leads
Please ensure that this readme file is always upto date with the latest helpful hints on how to help the new teammates 
get started with project. If there are new imports added to the list please ensure to add them.
If there are any other helpful ways to get the team started as quick as possible, please add a new section where 
is appropriate. Also, please ensure to update the contributors section, so everyone can be credited properly.

---
## Contributors

**Winter 2023**
- Vithurshan Umashanker 
- Samantha Yee
- Aaran Arulraj
- Nour Ziena
- Joanne Jo
- Gavin Shaw
- Shavaiz Khan
- Bowen Dou
- Dana Campbell

---
## License and copyright
### © Healthcare Systems R&A Inc.

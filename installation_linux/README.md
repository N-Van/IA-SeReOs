 To install the entire IA-SeReOs_env :

 1. You need a conda environnement
 2. Open a terminal (e.g. in Windows, Anaconda Powershell Prompt)
 3. Execute the IA_SeReOs_install.sh or .bat file 
 4. Activate the IA-SeReOs_env (conda activate IA-SeReOs_env)
 5. Run the IA_SeReOs_install_2.sh file

 If you can't run .sh files, verify the execution rights (run "chmod u+x nom_du_fichier.sh" to give those rights.)

To get all the repository :

git clone git@github.com:N-Van/IA-SeReOs.git

Then :

git branch -r | grep -v '\->' | while read remote; do git branch --track "${remote#origin/}" "$remote"; done
git fetch --all
git pull --all

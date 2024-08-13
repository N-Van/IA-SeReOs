 To install the entire IA-SeReOs_env :

 1. You need a conda environnement
 2. Go in a terminal
 3. Activate the conda environnement
 4. Run the IA_SeReOs_install.sh file
 5. Activate the IA-SeReOs_env (conda activate IA-SeReOs_env)
 6. Run the IA_SeReOs_install_2.sh file

 If you can't run .sh files, verify the execution rights (run "chmod u+x nom_du_fichier.sh" to give those rights.)

To get all the repository :

git clone git@github.com:N-Van/IA-SeReOs.git

Then :

git branch -r | grep -v '\->' | while read remote; do git branch --track "${remote#origin/}" "$remote"; done
git fetch --all
git pull --all

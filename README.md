# ToricRL

## Env

Actions är associerade med heltal på följande vis:

[upp: 0, ner: 1, vänster: 2, höger: 3]

Grundtillstånd:

  	0 (inga icketriviala loopar)
  	1 (vertikal icketrivial loop)
	2 (horisontell icketrivial loop)
	3 (vertikal + horisontell icketrivial loop)

# Git-goodies

Om man vill reverta till versionen på github (tar bort lokala varianter):

git reset --hard HEAD

git clean -f -d

git pull

##### Träna nätverk för att para ihop fel #####

För att träna nätverket används run_train, RL, Env och QNet. Syftet med dessa är att kunna
spara ett nätverk för att sedan använda det med MonteCarlo-Metod.

Med run_ready kan vi testa det sparade nätverket på ny data!

##### MonteCarlo - Metod #####
Här ska vi använda oss av runMC, RLMC, QNet och Env


Env.py ska fungera för både MonteCarlo och para ihop fel. Genom att specificera ett
extra input "checkgroundstate == True" för att ta hänsyn till groundstate.

# docker Hugging Face

Docker to run the Hugging Face environment, personalized to in parallel with turtlebot simulation

## HOW BUILD the docker

***Recommended:*** 
```
docker-compose build
```

## Graphics Terminal Permission

***Its necessary give permission to the terminal that will run the docker to have graphics interface access inside the docker***

```
xhost +
```

## HOW RUN the docker 

Make sure that u also cloned the turtlebot docker project to the same folder that was cloned this project (../)

***Recommended:***
```
docker compose run hugging_face bash
```

*or if you want to use the GPU Nvidia graphics (if your computer have a nvidia board)*

```
docker compose run hugging_face-nvidia bash
```

*If you are using windows run:*

```
docker compose run hugging_face-win bash
```

*or windows with nvidia*

```
docker compose run hugging_face-win-nvidia bash
```

## Script to run 

To run the script you need to have started the gazebo simulation and the navigation on the other docker already.

You also need to subsitute the tags of the initialize dockers inside the script file

so run :

```
. ./script_cl.sh
```




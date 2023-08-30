#!/bin/bash

sendpoint=true

while getopts "u:" flag
do
    case "${flag}" in
        u) sendpoint=${OPTARG};;
    esac
done

echo "Sendpoint: $sendpoint";


while :
do
    echo -e "\nWrite the text request: "
    read text
    echo "Text: $text"

    if [ "$text" == "rotate" ]; then
        echo "How many degrees I should rotate (clockwise): "
        read degrees

        docker exec -it e561e9f141e32d6a4b561f237eddb36c198079457acf64d0c19260cd9eaab1c1 /bin/bash -c "
                . /opt/ros/noetic/setup.bash && 
                . /home/rosuser/catkin_ws/devel/setup.bash && 
                export ROS_MASTER_URI=http://10.100.48.7:11311 &&
                rosrun beginner_tutorials navigation.py $degrees" 
    else

        docker exec -it e561e9f141e32d6a4b561f237eddb36c198079457acf64d0c19260cd9eaab1c1 /bin/bash -c "
            . /opt/ros/noetic/setup.bash && 
            . /home/rosuser/catkin_ws/devel/setup.bash && 
            export ROS_MASTER_URI=http://10.100.48.7:11311 &&
            rosrun beginner_tutorials listener.py" 

        docker exec -it dd01e84f8bcb2579d3df7380a874611c49cbd1536087585e6fa08984d79ae6f6 python /app/zero_shot.py $text
        theta=$?

        if [ $sendpoint == "true" ]; then
            echo "Sending Goal"
            docker exec -it e561e9f141e32d6a4b561f237eddb36c198079457acf64d0c19260cd9eaab1c1 /bin/bash -c "
                . /opt/ros/noetic/setup.bash && 
                . /home/rosuser/catkin_ws/devel/setup.bash && 
                export ROS_MASTER_URI=http://10.100.48.7:11311 &&
                rosrun beginner_tutorials gotoapoint.py $theta" 
        fi
    fi    

    

done

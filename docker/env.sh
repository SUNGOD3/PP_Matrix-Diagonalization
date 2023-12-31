#!/usr/bin/bash

Username=$(whoami)
ImageName=$(echo $Username"_img_dosm")
ContainerName=$(echo $Username"_container_dosm")

if [ $# -eq 0 ] ; then 
    echo "You need the parameter."
    echo "./env.sh --help for more infomation"
    exit 1
fi

if [ $1 == "start" ] ; then
    # create image
    if [ $(sudo docker images -a | grep -w $ImageName | wc -l) -eq 0  ] ; then
            echo "$ImageName not found, start to building '$ContainerName' container"
            sudo docker build -t $ImageName --build-arg UID=$(id -u) --build-arg USERNAME=$Username .
    fi
    # start the container
    if [ $(sudo docker ps -a | grep -w $ContainerName | wc -l) -eq 0 ] ; then # create the container
        echo "Run the '$ContainerName' container"
        sudo docker run -it \
            --gpus all \
            --name $ContainerName \
            -v $(pwd)/..:/home/$Username/workspace \
            $ImageName bash
    else
        if [ $(sudo docker ps | grep -w $ContainerName | wc -l) -eq 0 ] ; then # check whether the container has run already
            sudo docker start $ContainerName > /dev/null 2>&1
        fi
        sudo docker exec -it $ContainerName bash # get in the container
    fi
    
elif [ $1 == "stop" ] ; then # stop the container
    if ! sudo docker stop $ContainerName > /dev/null 2>&1 ; then
        echo "There is a unexpected wrong when stop the '$ContainerName'"
    elif [ $(sudo docker ps | grep -w $ContainerName | wc -l) -eq 0 ] ; then
        echo "The '$ContainerName' is not running."
    else
        echo "Stop the '$ContainerName' successfully!"
    fi
elif [ $1 == "rm" ] ; then # remove the container
    if ! sudo docker rm "$ContainerName" > /dev/null 2>&1 ; then
        echo "There is a unexpected wrong when removing '$ContainerName'"
    else
        echo "Remove the '$ContainerName' successfully!"
    fi
elif [ $1 == "rmi" ] ; then # remove the image
    if ! sudo docker rmi "$ImageName" > /dev/null 2>&1 ; then
        echo "There is a unexpected wrong when removing '$ImageName'"
    else
        echo "Remove the '$ImageName' successfully!"
    fi
elif [ $1 == "--help" ] ; then
    echo ./env "parameter"
    echo "start : to build or start the iamge and container"
    echo "stop : to stop the container"
    echo "rm : remove the conatiner"
    echo "rmi : remove the iamge"
else 
    echo "Wrong parameter!"
    echo "./env.sh --help for more infomation"
fi

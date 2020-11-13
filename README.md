# Application smartsolman_api

smartsolman est une application python qui recherche des similarites entre les messages solman

## Usage

Ce conteneur permet de lancer l'API de l'application smartsolman
A l'instanciation du conteneur, le serveur se lance automatiquement en tâche de fond

## Connection au serveur d'hébergement

Le déploiement actuel s'effectue sur la VM 178.32.116.200 
User : sapadm
Password: idem que sapdocpy

Une fois connecté, basculer en root depuis un terminal :
```bash
su -
```
Password: idem que root de sapdocpy.pasapas.com

## Connection à Docker

Créer votre compte Docker sur la plateforme [Docker Hub](https://hub.docker.com/) 

Poursuivez depuis un terminal à l'emplacement de l'application /smartsolman/smarsolman_api
```bash
docker login
```
Saissisez votre Compte et Mot_de_passe Docker

## Installation

Pour créer l'image, executez la commande suivante

```bash
docker build -f Dockerfile -t smartsol:api .
```

## Deploiement

Pour lancer le service, creez un conteneur depuis l'image avec la commande suivante

```bash
docker run --mount type=bind,source=/home/sapadm/volume,target=/smartsolman/data --sysctl net.ipv4.ip_forward=1 -d -p 5000:5000 <image_id>
```
Cette commande va :
- Instancier le conteneur et lancer directement le service en tâche de fond
- Lier un volume du conteneur à un espace disque du serveur pour la bonne persistance des données
- Rediriger les écoutes de port de l'ipv6 vers l'ipv4, ce qui permettra de stabiliser les appels client à l'API

(to remind --sysctl net.ipv6.conf.all.forwarding=1)

## Tester le modèle

Il n'y a pas encore d'adresse public au service. Il est consultable depuis le localhost

Pour tester le service, se positionner dans app/dev/full_model depuis le terminal

Exécuter le script cli.py

Un message pré-enregistré est envoyé au service. Pas de sélection de message test possible pour le moment

## Commandes docker utiles

Consulter les images
```bash
docker images
```
Consulter les conteneurs
```bash
docker ps -a
```
Supprimer un conteneur
```bash
docker container stop <container_id>
docker container rm <container_id>
```
Supprimer une image
```bash
docker rmi <image_id>
```
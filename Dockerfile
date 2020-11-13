# telecharge l'image de l'OS depuis docker_hub
FROM opensuse/leap:15

# copie l'application dans l'image
COPY . /smartsolman/

# definit le repertoire courant
WORKDIR /smartsolman

# installe le contexte
RUN zypper install -y python3 gcc python3-devel python3-pip sqlite3 
RUN pip install -r requirements.txt

# ouvre le port du service
EXPOSE 5000

# lance l'application a la creation du conteneur
CMD ["/smartsolman/app/dev/full_model/server_DF.py"]
ENTRYPOINT [ "python3" ]

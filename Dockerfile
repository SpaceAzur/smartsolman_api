FROM opensuse/leap:15

COPY . /smartsolman/

# VOLUME /smartsolman/data

WORKDIR /smartsolman

RUN zypper install -y python3 gcc python3-devel python3-pip sqlite3 
RUN pip install -r requirements.txt

# EXPOSE 5000

# CMD ["/smartsolman/app/dev/full_model/server_DF.py"]
# ENTRYPOINT [ "python3" ]

database="exodus"
username="exodus"
password="exodus"
mongo -- $database <<EOF
  db = db.getSiblingDB('$database')
  db.dropUser('$username')
  db.createUser({
    user: '$username',
    pwd: '$password',
    roles: [
      {
        role: 'readWrite',
        db: '$database',
      }
    ],
    mechanisms : ['SCRAM-SHA-1']
  })
EOF

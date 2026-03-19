#### Environment
- redis-stack
```sh
brew tap redis-stack/redis-stack
brew install redis-stack
alias redis-stack="/opt/homebrew/bin/redis-stack-server /opt/homebrew/etc/redis-stack.conf"
printf '%s\n' "requirepass LOCALDEV" | sudo tee -a /opt/homebrew/etc/redis-stack.conf
```

#### Running:
```sh
python ai.py
```

#### Usage:
```sh
./ask.sh "Where can I find cost reduction guidance?"
```

#### Updates:
- First put new doc files in `./chunker/docs` folder, then from inside the `chunker/` directory, run:
```sh
for DOC in `ls docs`; do python chunker.py "./docs/$DOC"; done
```

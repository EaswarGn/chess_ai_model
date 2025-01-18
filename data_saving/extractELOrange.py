import argparse
import bz2
import re
import zstandard as zstd

low_time_re = re.compile(r'(\d+\. )?\S+ \{ \[%clk 0:00:[210]\d\]')
moveRegex = re.compile(r'\d+[.][ \.](\S+) (?:{[^}]*} )?(\S+)')

#@haibrid_chess_utils.logged_main
def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('eloMin', type=int, help='min ELO')
    parser.add_argument('eloMax', type=int, help='max ELO')
    parser.add_argument('output', help='output file')
    parser.add_argument('targets', nargs='+', help='target files')
    parser.add_argument('--remove_bullet', action='store_true', help='Remove bullet and ultrabullet games')
    parser.add_argument('--remove_low_time', action='store_true', help='Remove low time moves from games')

    args = parser.parse_args()
    gamesWritten = 0
    print(f"Starting writing to: {args.output}")

    with open(args.output, 'wb') as fout:  # Open the output file in binary write mode
        compressor = zstd.ZstdCompressor()
        with compressor.stream_writer(fout) as f:
            for num_files, target in enumerate(sorted(args.targets)):
                print(f"{num_files} reading: {target}")
                Games = LightGamesFile(target, parseMoves=False)
                for i, (dat, lines) in enumerate(Games):
                    try:
                        whiteELO = int(dat['WhiteElo'])
                        BlackELO = int(dat['BlackElo'])
                    except ValueError:
                        continue
                    if whiteELO > args.eloMax or whiteELO <= args.eloMin:
                        continue
                    elif BlackELO > args.eloMax or BlackELO <= args.eloMin:
                        continue
                    elif dat['Result'] not in ['1-0', '0-1', '1/2-1/2']:
                        continue
                    elif args.remove_bullet and 'Bullet' in dat['Event']:
                        continue
                    else:
                        if args.remove_low_time:
                            f.write(remove_low_time(lines).encode('utf-8'))  # Encode to bytes
                        else:
                            f.write(lines.encode('utf-8'))  # Encode to bytes
                        gamesWritten += 1
                    if i % 1000 == 0:
                        print(f"{i}: written {gamesWritten} files {num_files}: {target}".ljust(79), end='\r')
                print(f"Done: {target} {i}".ljust(79))

            
def remove_low_time(g_str):
    r = low_time_re.search(g_str)
    if r is None:
        return g_str
    end = g_str[-20:].split(' ')[-1]
    return g_str[:r.span()[0]] + end


class LightGamesFile(object):
    def __init__(self, path, parseMoves = True, just_games = False):
        if path.endswith('bz2'):
            self.f = bz2.open(path, 'rt')
        elif path.endswith('zst'):
            dctx = zstd.ZstdDecompressor()
            self.f = dctx.stream_reader(open(path, 'rb'))
            self.f = open(self.f, 'rt')  # Convert binary stream to text stream
        else:
            self.f = open(path, 'r')
        self.parseMoves = parseMoves
        self.just_games = just_games
        self._peek = None

    def __iter__(self):
        try:
            while True:
                yield self.readNextGame()
        except StopIteration:
            return

    def peekNextGame(self):
        if self._peek is None:
            self._peek = self.readNextGame()
        return self._peek

    def readNextGame(self):
        #self.f.readline()
        if self._peek is not None:
            g = self._peek
            self._peek = None
            return g
        ret = {}
        lines = ''
        if self.just_games:
            first_hit = False
            for l in self.f:
                lines += l
                if len(l) < 2:
                    if first_hit:
                        break
                    else:
                        first_hit = True
        else:
            for l in self.f:
                lines += l
                if len(l) < 2:
                    if len(ret) >= 2:
                        break
                    else:
                        raise RuntimeError(l)
                else:
                    k, v, _ = l.split('"')
                    ret[k[1:-1]] = v
            nl = self.f.readline()
            lines += nl
            if self.parseMoves:
                ret['moves'] = re.findall(moveRegex, nl)
            lines += self.f.readline()
        if len(lines) < 1:
            raise StopIteration
        return ret, lines

    def readBatch(self, n):
        ret = []
        for i in range(n):
            try:
                ret.append(self.readNextGame())
            except StopIteration:
                break
        return ret

    def getWinRates(self, extraKey = None):
        # Assumes same players in all games
        dat, _ = self.peekNextGame()
        p1, p2 = sorted((dat['White'], dat['Black']))
        d = {
            'name' : f"{p1} v {p2}",
            'p1' : p1,
            'p2' : p2,
            'games' : 0,
            'wins' : 0,
            'ties' : 0,
            'losses' : 0,
            }
        if extraKey is not None:
            d[extraKey] = {}
        for dat, _ in self:
            d['games'] += 1
            if extraKey is not None and dat[extraKey] not in d[extraKey]:
                d[extraKey][dat[extraKey]] = []
            if p1 == dat['White']:
                if dat['Result'] == '1-0':
                    d['wins'] += 1
                    if extraKey is not None:
                        d[extraKey][dat[extraKey]].append(1)
                elif dat['Result'] == '0-1':
                    d['losses'] += 1
                    if extraKey is not None:
                        d[extraKey][dat[extraKey]].append(0)
                else:
                    d['ties'] += 1
                    if extraKey is not None:
                        d[extraKey][dat[extraKey]].append(.5)
            else:
                if dat['Result'] == '0-1':
                    d['wins'] += 1
                    if extraKey is not None:
                        d[extraKey][dat[extraKey]].append(1)
                elif dat['Result'] == '1-0':
                    d['losses'] += 1
                    if extraKey is not None:
                        d[extraKey][dat[extraKey]].append(0)
                else:
                    d['ties'] += 1
                    if extraKey is not None:
                        d[extraKey][dat[extraKey]].append(.5)
        return d

    def __del__(self):
        try:
            self.f.close()
        except AttributeError:
            pass

if __name__ == '__main__':
    main()

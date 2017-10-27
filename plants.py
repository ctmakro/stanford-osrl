# obstaclex, height, radius

csv = ''
with open('sublog.csv','r') as f:
    csv = f.read()

csv = csv.split('\n')
csv = [l.strip() for l in csv]
csv = list(filter(lambda x:len(x)>0,csv))

print(len(csv),'lines')

csv = [[float(k) for k in l.split(',')] for l in csv]

lastabsdist = 0.
lastpsoas = -4.
tacos = []
psoas = []
for l in csv:
    px = l[1]

    lpsoas,rpsoas = l[36],l[37]

    if abs(lpsoas-lastpsoas)>1e-8:
        lastpsoas = lpsoas
        psoas.append([lpsoas,rpsoas])

    bdist = l[38]
    bheight = l[39]
    bradius = l[40]

    if bdist == 100:
        continue

    absdist = bdist+px
    if abs(absdist - lastabsdist)>1e-8:
        lastabsdist = absdist

        tacos.append([absdist,bheight,bradius])

print(tacos,psoas)

with open('rosetta.py','w') as f:
    f.write('tacos=')
    f.write(str(tacos))
    f.write('\npsoas=')
    f.write(str(psoas))

# Opgaver
# For et 8-bit gråtonebillede af størrelsen 512x512 pixels, hvor mange forskellige billeder kan man lave?
#256^(512x512)  # Mange lol
        # Hvad hvis det var et RGB billede?
            # WTF mange flere = 3^(256^(512x512))
import cv2

print("I et 100x100 pixel gråtonebillede er hver pixel repræsenteret med 256 gråtoneværdier. Hvor meget hukommelse (bytes) er krævet for at gemme dette billede?")
print(f'svar = 100 x 100 = {100*100} \n')

print('Et RGB billede konverteres til gråtonebillede. Under konverteringen sættes WB=0 og de to resterende farvekanaler vægtes lige.')
print("Vi kigger nu på en pixel i gråtonebilledet har værdien 100. Hvad var den grønne værdi i den tilsvarende RGB pixel, hvis vi ved at R=20?")
print(f'svar = {80} \n')

print("Hvordan er farver repræsenteret i HTML?")
print("RGB,HEX,HSL, RGBA or HSLA\n")
print(f'Konverter en RGB pixel med værdien (R,G,B) = (20,40,60) til (r,g,b), (r,g,I), (H,S,L) og HEX repræsentation.')
R=20
G=40
B=60
print(f'RGB = ({R},{G},{B})')
print(f'rgb = ({R/(R+G+B)},{G/(R+G+B)},{B/(R+G+B)}')
print(f'rgI = ({R/(R+G+B)},{G/(R+G+B)},{(R+G+B)/3})')
print(f'HSL = (210, 50, 15.7)')
V=(100)
print(f'HSI = ({(B-R)/(V-0),1,100})')
print(f'HEX = #14283C')

# Download og installer ImageJ (https://imagej.nih.gov/ij/download.html)
# Load et RGB billede

# Kig på pixelværdierne og leg med funktionerne under 'Image->Color'. F.eks. brug 'Split channels' og 'Make composite' for at lære mere om kompositionen af RGB farver
# Leg med farvethreshold-funktionen 'Image->Adjust->Color threshold'. Lær hvordan du kan segmentere én farve i billedet. Prøv både HSB og RGB farvemodeller.
# Download og installer OpenCV med dit favoritsprog og IDE.
package rs.ac.bg.etf.players;

public class Igrac extends Player {

	private String tip = null;

	@Override
	public Move getNextMove() {
		if (opponentMoves.isEmpty())
			return Move.DONTPUTCOINS;
		if (tip == null || tip == "staviliJedan" || tip == "staviliDva" || tip == "staviliNula") {
			playerType();
		}
		return playMove();
	}

	private void playerType() {

		if (opponentMoves.size() < 2) {
			if (opponentMoves.get(0) == Move.PUT1COIN)
				tip = "staviliJedan";
			else if (opponentMoves.get(0) == Move.PUT2COINS)
				tip = "staviliDva";
			else
				tip = "staviliNula";
			return;
		}

		if (opponentMoves.get(0) == Move.DONTPUTCOINS && opponentMoves.get(1) == Move.DONTPUTCOINS) {
			tip = "Stinger";
			return;
		}

		if (opponentMoves.get(0) == Move.DONTPUTCOINS && opponentMoves.get(1) == Move.PUT1COIN) {
			tip = "Igrac";
			return;
		}

		if (opponentMoves.get(0) == Move.PUT1COIN && opponentMoves.get(1) == Move.DONTPUTCOINS) {
			tip = "CopyCat";
			return;
		}
		if (opponentMoves.get(0) == Move.PUT1COIN && opponentMoves.get(1) == Move.PUT1COIN) {
			tip = "Forgiver";
			return;
		}
		if (opponentMoves.get(0) == Move.PUT2COINS && opponentMoves.get(1) == Move.PUT2COINS) {
			tip = "Goody";
			return;
		}
		if (opponentMoves.get(0) == Move.PUT2COINS && opponentMoves.get(1) == Move.DONTPUTCOINS) {
			tip = "Avenger";
			return;
		}

		if (opponentMoves.size() == 2 && !(tip.equals("Avenger")) && !(tip.equals("Goody")) && !(tip.equals("Forgiver"))
				&& !(tip.equals("CopyCat")) && !(tip.equals("Igrac")) && !(tip.equals("Stinger"))) {
			tip = "nepoznat";
			return;
		}

	}

	private Move playMove() {
		switch (tip) {
		case ("staviliJedan"):
			return Move.PUT1COIN;
		case ("staviliDva"):
			return Move.DONTPUTCOINS;
		case ("staviliNula"):
			return Move.PUT1COIN;
		case ("Igrac"):
			return Move.PUT2COINS;
		case ("Goody"):
			return Move.DONTPUTCOINS;
		case ("Stinger"):
			return Move.DONTPUTCOINS;
		case ("CopyCat"):
			return Move.PUT1COIN;
		case ("Forgiver"):
			return Move.PUT1COIN;
		case ("Avenger"):
			return Move.DONTPUTCOINS;
		default:
			return Move.DONTPUTCOINS;
		}
	}

	@Override
	public void resetPlayerState() {
		super.resetPlayerState();
		tip = null;
	}

}
